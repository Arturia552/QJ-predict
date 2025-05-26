use anyhow::Result;
use chrono::{DateTime, Utc};
use polars::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

// ========== 核心数据结构 ==========

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceDataPoint {
    pub device_id: String,
    pub timestamp: DateTime<Utc>,
    pub pressure: f64,
    pub current: f64,
    pub flow_rate: f64,
    pub temperature: f64,
    pub voltage: f64,
    pub sinking_degree: f64,
}

#[derive(Debug, Clone)]
pub struct AnalysisResult {
    pub device_id: String,
    pub parameter: String,
    pub warning_type: WarningType,
    pub severity: Severity,
    pub current_value: f64,
    pub score: f64,
    pub probability: f64,
    pub message: String,
    pub recommendation: String,
    pub timestamp: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub enum WarningType {
    ThresholdViolation,
    StatisticalAnomaly,
    TrendAlert,
    RapidChange,
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum Severity {
    Info,
    Low,
    Medium,
    High,
    Critical,
}

// ========== Polars数据管理器 ==========

/// 基于Polars的高性能数据存储和处理管理器
pub struct PolarDataManager {
    /// 设备数据存储 (使用LazyFrame进行惰性计算)
    device_data: Arc<RwLock<HashMap<String, LazyFrame>>>,
    /// 数据保留策略
    retention_config: DataRetentionConfig,
}

#[derive(Debug, Clone)]
pub struct DataRetentionConfig {
    pub max_rows_per_device: usize,
    pub max_days: i64,
}

impl PolarDataManager {
    pub fn new(retention_config: DataRetentionConfig) -> Self {
        Self {
            device_data: Arc::new(RwLock::new(HashMap::new())),
            retention_config,
        }
    }

    /// 批量添加设备数据 (高性能批处理)
    pub fn add_batch_data(
        &self,
        device_id: &str,
        data_points: Vec<DeviceDataPoint>,
    ) -> PolarsResult<()> {
        if data_points.is_empty() {
            return Ok(());
        }

        // 转换为Polars DataFrame
        let df = self.create_dataframe_from_points(&data_points)?;
        let new_lazy = df.lazy();

        let mut data_store = self.device_data.write().unwrap();

        if let Some(existing_lazy) = data_store.get(device_id) {
            let combined = concat([existing_lazy.clone(), new_lazy], UnionArgs::default())?
                .sort(["timestamp"], SortMultipleOptions::default())
                .with_row_index("row_id", None);

            let retained = self.apply_retention_policy(combined)?;
            data_store.insert(device_id.to_string(), retained);
        } else {
            data_store.insert(device_id.to_string(), new_lazy);
        }

        Ok(())
    }

    /// 获取设备的历史数据 (惰性计算优化)
    pub fn get_device_data(&self, device_id: &str) -> Option<LazyFrame> {
        let data_store = self.device_data.read().unwrap();
        data_store.get(device_id).cloned()
    }

    /// 获取指定参数的时间序列数据
    pub fn get_parameter_timeseries(
        &self,
        device_id: &str,
        parameter: &str,
        window_size: usize,
    ) -> PolarsResult<DataFrame> {
        let data_store = self.device_data.read().unwrap();
        let lazy_df = data_store
            .get(device_id)
            .ok_or_else(|| PolarsError::ComputeError("Device not found".into()))?;

        lazy_df
            .clone()
            .select([col("timestamp"), col(parameter)])
            .tail(window_size as u32)
            .collect()
    }

    /// 使用Polars计算统计指标
    pub fn calculate_statistics(
        &self,
        device_id: &str,
        parameter: &str,
        window_size: usize,
    ) -> PolarsResult<StatisticalSummary> {
        let data_store = self.device_data.read().unwrap();
        let lazy_df = data_store
            .get(device_id)
            .ok_or_else(|| PolarsError::ComputeError("Device not found".into()))?;

        let stats = lazy_df
            .clone()
            .select([col(parameter)])
            .tail(window_size as u32)
            .select([
                col(parameter).mean().alias("mean"),
                col(parameter).std(1).alias("std_dev"),
                col(parameter).median().alias("median"),
                col(parameter)
                    .quantile(lit(0.25), QuantileMethod::Linear)
                    .alias("q1"),
                col(parameter)
                    .quantile(lit(0.75), QuantileMethod::Linear)
                    .alias("q3"),
                col(parameter).min().alias("min"),
                col(parameter).max().alias("max"),
                col(parameter).count().alias("count"),
            ])
            .collect()?;
        if stats.height() == 0 {
            return Err(PolarsError::ComputeError("No data available".into()));
        }

        Ok(StatisticalSummary {
            mean: stats.column("mean")?.f64()?.get(0).unwrap_or(0.0),
            std_dev: stats.column("std_dev")?.f64()?.get(0).unwrap_or(0.0),
            median: stats.column("median")?.f64()?.get(0).unwrap_or(0.0),
            q1: stats.column("q1")?.f64()?.get(0).unwrap_or(0.0),
            q3: stats.column("q3")?.f64()?.get(0).unwrap_or(0.0),
            min: stats.column("min")?.f64()?.get(0).unwrap_or(0.0),
            max: stats.column("max")?.f64()?.get(0).unwrap_or(0.0),
            count: stats.column("count")?.u32()?.get(0).unwrap_or(0) as usize,
        })
    }

    fn create_dataframe_from_points(&self, points: &[DeviceDataPoint]) -> PolarsResult<DataFrame> {
        let timestamps: Vec<i64> = points
            .iter()
            .map(|p| p.timestamp.timestamp_millis()) // 改用毫秒，避免微秒可能的溢出
            .collect();

        let device_ids: Vec<String> = points.iter().map(|p| p.device_id.clone()).collect();
        let pressures: Vec<f64> = points.iter().map(|p| p.pressure).collect();
        let currents: Vec<f64> = points.iter().map(|p| p.current).collect();
        let flow_rates: Vec<f64> = points.iter().map(|p| p.flow_rate).collect();
        let temperatures: Vec<f64> = points.iter().map(|p| p.temperature).collect();
        let voltages: Vec<f64> = points.iter().map(|p| p.voltage).collect();
        let sinking_degrees: Vec<f64> = points.iter().map(|p| p.sinking_degree).collect();

        df!(
            "timestamp" => timestamps,
            "device_id" => device_ids,
            "pressure" => pressures,
            "current" => currents,
            "flow_rate" => flow_rates,
            "temperature" => temperatures,
            "voltage" => voltages,
            "sinking_degree" => sinking_degrees
        )
    }

    fn apply_retention_policy(&self, lazy_df: LazyFrame) -> PolarsResult<LazyFrame> {
        let cutoff_timestamp = (Utc::now()
            - chrono::Duration::days(self.retention_config.max_days))
        .timestamp_millis(); // 改用毫秒保持一致

        Ok(lazy_df
            .filter(col("timestamp").gt_eq(lit(cutoff_timestamp)))
            .tail(self.retention_config.max_rows_per_device as u32))
    }
}

#[derive(Debug, Clone)]
pub struct StatisticalSummary {
    pub mean: f64,
    pub std_dev: f64,
    pub median: f64,
    pub q1: f64,
    pub q3: f64,
    pub min: f64,
    pub max: f64,
    pub count: usize,
}

// ========== 1. Polars优化的阈值分析器 ==========

/// 使用Polars批量处理的阈值分析器
pub struct PolarsThresholdAnalyzer {
    data_manager: Arc<PolarDataManager>,
    thresholds: HashMap<String, ParameterThreshold>,
    name: String,
}

#[derive(Debug, Clone)]
pub struct ParameterThreshold {
    pub critical_min: Option<f64>,
    pub warning_min: Option<f64>,
    pub warning_max: Option<f64>,
    pub critical_max: Option<f64>,
}

impl PolarsThresholdAnalyzer {
    pub fn new(name: String, data_manager: Arc<PolarDataManager>) -> Self {
        Self {
            data_manager,
            thresholds: HashMap::new(),
            name,
        }
    }

    pub fn configure_oil_defaults(&mut self) {
        self.thresholds.insert(
            "pressure".to_string(),
            ParameterThreshold {
                critical_min: Some(0.5),
                warning_min: Some(1.0),
                warning_max: Some(4.0),
                critical_max: Some(5.0),
            },
        );

        self.thresholds.insert(
            "current".to_string(),
            ParameterThreshold {
                critical_min: Some(5.0),
                warning_min: Some(8.0),
                warning_max: Some(25.0),
                critical_max: Some(30.0),
            },
        );

        self.thresholds.insert(
            "flow_rate".to_string(),
            ParameterThreshold {
                critical_min: Some(5.0),
                warning_min: Some(10.0),
                warning_max: Some(50.0),
                critical_max: Some(60.0),
            },
        );

        self.thresholds.insert(
            "temperature".to_string(),
            ParameterThreshold {
                critical_min: Some(-10.0),
                warning_min: Some(0.0),
                warning_max: Some(80.0),
                critical_max: Some(100.0),
            },
        );
    }

    /// 批量阈值检测 (利用Polars向量化计算)
    pub fn batch_analyze(
        &self,
        device_id: &str,
        batch_size: usize,
    ) -> PolarsResult<Vec<AnalysisResult>> {
        let lazy_df = self
            .data_manager
            .get_device_data(device_id)
            .ok_or_else(|| PolarsError::ComputeError("Device not found".into()))?;

        let recent_data = lazy_df.tail(batch_size as u32).collect()?;

        let mut results = Vec::new();
        let parameters = ["pressure", "current", "flow_rate", "temperature"];

        for parameter in parameters {
            if let Some(threshold) = self.thresholds.get(parameter) {
                let violations =
                    self.detect_violations_vectorized(&recent_data, parameter, threshold)?;
                results.extend(violations);
            }
        }

        Ok(results)
    }

    fn detect_violations_vectorized(
        &self,
        df: &DataFrame,
        parameter: &str,
        threshold: &ParameterThreshold,
    ) -> PolarsResult<Vec<AnalysisResult>> {
        let mut results = Vec::new();

        // 使用Polars表达式进行向量化阈值检测
        let mut conditions = Vec::new();
        let mut violation_types = Vec::new();

        if let Some(critical_max) = threshold.critical_max {
            conditions.push(col(parameter).gt_eq(lit(critical_max)));
            violation_types.push(("critical_max", critical_max, Severity::Critical));
        }

        if let Some(critical_min) = threshold.critical_min {
            conditions.push(col(parameter).lt_eq(lit(critical_min)));
            violation_types.push(("critical_min", critical_min, Severity::Critical));
        }

        if let Some(warning_max) = threshold.warning_max {
            conditions.push(col(parameter).gt_eq(lit(warning_max)));
            violation_types.push(("warning_max", warning_max, Severity::High));
        }

        if let Some(warning_min) = threshold.warning_min {
            conditions.push(col(parameter).lt_eq(lit(warning_min)));
            violation_types.push(("warning_min", warning_min, Severity::High));
        }

        for (condition, (violation_type, limit, severity)) in
            conditions.iter().zip(violation_types.iter())
        {
            let violations = df
                .clone()
                .lazy()
                .filter(condition.clone())
                .select([col("device_id"), col("timestamp"), col(parameter)])
                .collect()?;

            for row_idx in 0..violations.height() {
                let device_id = violations
                    .column("device_id")?
                    .str()?
                    .get(row_idx)
                    .unwrap_or("")
                    .to_string();
                let timestamp_i64 = violations
                    .column("timestamp")?
                    .i64()?
                    .get(row_idx)
                    .unwrap_or(0);
                let timestamp =
                    DateTime::from_timestamp_millis(timestamp_i64).unwrap_or(Utc::now()); // 改用毫秒转换
                let value = violations
                    .column(parameter)?
                    .f64()?
                    .get(row_idx)
                    .unwrap_or(0.0);

                let mut metadata = HashMap::new();
                metadata.insert("analyzer".to_string(), self.name.clone());
                metadata.insert("violation_type".to_string(), violation_type.to_string());
                metadata.insert("threshold_value".to_string(), limit.to_string());

                results.push(AnalysisResult {
                    device_id,
                    parameter: parameter.to_string(),
                    warning_type: WarningType::ThresholdViolation,
                    severity: severity.clone(),
                    current_value: value,
                    score: ((value - limit) / limit * 100.0).abs(),
                    probability: 1.0,
                    message: format!("{}违反{}阈值: {:.2}", parameter, violation_type, limit),
                    recommendation: self.get_threshold_recommendation(violation_type, parameter),
                    timestamp,
                    metadata,
                });
            }
        }

        Ok(results)
    }

    fn get_threshold_recommendation(&self, violation_type: &str, parameter: &str) -> String {
        match violation_type {
            "critical_max" => format!("{}超过临界最大值，立即停机检查！", parameter),
            "critical_min" => format!("{}低于临界最小值，检查设备运行状态！", parameter),
            "warning_max" => format!("{}接近最大限值，密切监控", parameter),
            "warning_min" => format!("{}接近最小限值，检查供液情况", parameter),
            _ => format!("{}异常，需要关注", parameter),
        }
    }
}

// ========== 2. Polars优化的统计异常分析器 ==========

/// 使用Polars统计函数的异常检测器
pub struct PolarsStatisticalAnalyzer {
    data_manager: Arc<PolarDataManager>,
    config: StatisticalConfig,
    name: String,
}

#[derive(Debug, Clone)]
pub struct StatisticalConfig {
    pub sigma_threshold: f64,
    pub confidence_level: f64,
    pub min_samples: usize,
    pub window_size: usize,
}

impl PolarsStatisticalAnalyzer {
    pub fn new(
        name: String,
        data_manager: Arc<PolarDataManager>,
        config: StatisticalConfig,
    ) -> Self {
        Self {
            data_manager,
            config,
            name,
        }
    }

    /// 使用Polars进行高效的统计异常检测
    pub fn analyze_statistical_anomalies(
        &self,
        device_id: &str,
        parameter: &str,
        current_value: f64,
        timestamp: DateTime<Utc>,
    ) -> PolarsResult<Option<AnalysisResult>> {
        let stats = self.data_manager.calculate_statistics(
            device_id,
            parameter,
            self.config.window_size,
        )?;

        if stats.count < self.config.min_samples {
            return Ok(None);
        }

        // Z-Score检测
        if let Some(z_result) =
            self.detect_z_score_anomaly(&stats, current_value, device_id, parameter, timestamp)?
        {
            return Ok(Some(z_result));
        }

        // IQR检测
        if let Some(iqr_result) =
            self.detect_iqr_anomaly(&stats, current_value, device_id, parameter, timestamp)?
        {
            return Ok(Some(iqr_result));
        }

        Ok(None)
    }

    fn detect_z_score_anomaly(
        &self,
        stats: &StatisticalSummary,
        current_value: f64,
        device_id: &str,
        parameter: &str,
        timestamp: DateTime<Utc>,
    ) -> PolarsResult<Option<AnalysisResult>> {
        if stats.std_dev <= 0.0 {
            return Ok(None);
        }

        let z_score = (current_value - stats.mean) / stats.std_dev;

        if z_score.abs() > self.config.sigma_threshold {
            let severity = match z_score.abs() {
                s if s > 5.0 => Severity::Critical,
                s if s > 4.0 => Severity::High,
                s if s > 3.0 => Severity::Medium,
                _ => Severity::Low,
            };

            let mut metadata = HashMap::new();
            metadata.insert("analyzer".to_string(), self.name.clone());
            metadata.insert("method".to_string(), "z_score".to_string());
            metadata.insert("mean".to_string(), format!("{:.3}", stats.mean));
            metadata.insert("std_dev".to_string(), format!("{:.3}", stats.std_dev));
            metadata.insert("sample_size".to_string(), stats.count.to_string());

            return Ok(Some(AnalysisResult {
                device_id: device_id.to_string(),
                parameter: parameter.to_string(),
                warning_type: WarningType::StatisticalAnomaly,
                severity,
                current_value,
                score: z_score,
                probability: self.calculate_z_probability(z_score),
                message: format!("{}出现{:.1}σ统计异常", parameter, z_score.abs()),
                recommendation: format!(
                    "当前值偏离历史均值{:.1}个标准差，建议检查设备",
                    z_score.abs()
                ),
                timestamp,
                metadata,
            }));
        }

        Ok(None)
    }

    fn detect_iqr_anomaly(
        &self,
        stats: &StatisticalSummary,
        current_value: f64,
        device_id: &str,
        parameter: &str,
        timestamp: DateTime<Utc>,
    ) -> PolarsResult<Option<AnalysisResult>> {
        let iqr = stats.q3 - stats.q1;
        if iqr <= 0.0 {
            return Ok(None);
        }

        let lower_fence = stats.q1 - 1.5 * iqr;
        let upper_fence = stats.q3 + 1.5 * iqr;

        if current_value < lower_fence || current_value > upper_fence {
            let score = if current_value < lower_fence {
                (lower_fence - current_value) / iqr
            } else {
                (current_value - upper_fence) / iqr
            };

            let severity = match score {
                s if s > 3.0 => Severity::Critical,
                s if s > 2.0 => Severity::High,
                s if s > 1.0 => Severity::Medium,
                _ => Severity::Low,
            };

            let mut metadata = HashMap::new();
            metadata.insert("analyzer".to_string(), self.name.clone());
            metadata.insert("method".to_string(), "iqr".to_string());
            metadata.insert("q1".to_string(), format!("{:.3}", stats.q1));
            metadata.insert("q3".to_string(), format!("{:.3}", stats.q3));
            metadata.insert("iqr".to_string(), format!("{:.3}", iqr));

            return Ok(Some(AnalysisResult {
                device_id: device_id.to_string(),
                parameter: parameter.to_string(),
                warning_type: WarningType::StatisticalAnomaly,
                severity,
                current_value,
                score,
                probability: 0.007,
                message: format!("{}超出IQR正常范围", parameter),
                recommendation: format!("当前值超出四分位距正常范围{:.1}倍", score),
                timestamp,
                metadata,
            }));
        }

        Ok(None)
    }

    fn calculate_z_probability(&self, z_score: f64) -> f64 {
        // 使用标准正态分布计算概率
        use statrs::distribution::{ContinuousCDF, Normal};
        let normal = Normal::new(0.0, 1.0).unwrap();
        if z_score > 0.0 {
            1.0 - normal.cdf(z_score.abs())
        } else {
            normal.cdf(-z_score.abs())
        }
    }
}

// ========== 3. Polars优化的趋势分析器 ==========

/// 使用Polars窗口函数的趋势分析器
pub struct PolarsTrendAnalyzer {
    data_manager: Arc<PolarDataManager>,
    config: TrendConfig,
    name: String,
}

#[derive(Debug, Clone)]
pub struct TrendConfig {
    pub window_size: usize,
    pub min_points: usize,
    pub slope_threshold: f64,
    pub r_squared_threshold: f64,
}

impl PolarsTrendAnalyzer {
    pub fn new(name: String, data_manager: Arc<PolarDataManager>, config: TrendConfig) -> Self {
        Self {
            data_manager,
            config,
            name,
        }
    }

    /// 使用Polars进行高效的趋势分析
    pub fn analyze_trend(
        &self,
        device_id: &str,
        parameter: &str,
        current_value: f64,
        timestamp: DateTime<Utc>,
    ) -> PolarsResult<Option<AnalysisResult>> {
        let timeseries = self.data_manager.get_parameter_timeseries(
            device_id,
            parameter,
            self.config.window_size,
        )?;

        if timeseries.height() < self.config.min_points {
            return Ok(None);
        }

        // 使用Polars计算线性回归
        let trend_result = self.calculate_trend_with_polars(&timeseries, parameter)?;

        if trend_result.r_squared < self.config.r_squared_threshold {
            return Ok(None);
        }

        let is_significant = trend_result.slope.abs() > self.config.slope_threshold;
        if !is_significant {
            return Ok(None);
        }

        let severity = match trend_result.slope.abs() {
            s if s > 0.05 => Severity::High,
            s if s > 0.02 => Severity::Medium,
            _ => Severity::Low,
        };

        let mut metadata = HashMap::new();
        metadata.insert("analyzer".to_string(), self.name.clone());
        metadata.insert("slope".to_string(), format!("{:.6}", trend_result.slope));
        metadata.insert(
            "r_squared".to_string(),
            format!("{:.3}", trend_result.r_squared),
        );
        metadata.insert("data_points".to_string(), timeseries.height().to_string());

        Ok(Some(AnalysisResult {
            device_id: device_id.to_string(),
            parameter: parameter.to_string(),
            warning_type: WarningType::TrendAlert,
            severity,
            current_value,
            score: trend_result.slope,
            probability: trend_result.r_squared,
            message: format!("{}检测到显著趋势变化", parameter),
            recommendation: self.generate_trend_recommendation(trend_result.slope, parameter),
            timestamp,
            metadata,
        }))
    }

    fn calculate_trend_with_polars(
        &self,
        df: &DataFrame,
        parameter: &str,
    ) -> PolarsResult<TrendResult> {
        // 使用Polars进行线性回归计算
        let with_index = df
            .clone()
            .lazy()
            .with_row_index("x", None)
            .with_columns([col("x").cast(DataType::Float64), col(parameter).alias("y")])
            .collect()?;

        let n = with_index.height() as f64;
        let stats = with_index
            .clone()
            .lazy()
            .select([
                col("x").sum().alias("sum_x"),
                col("y").sum().alias("sum_y"),
                (col("x") * col("y")).sum().alias("sum_xy"),
                (col("x").pow(lit(2))).sum().alias("sum_x2"),
                (col("y").pow(lit(2))).sum().alias("sum_y2"),
                col("y").mean().alias("mean_y"),
            ])
            .collect()?;

        let sum_x = stats.column("sum_x")?.f64()?.get(0).unwrap_or(0.0);
        let sum_y = stats.column("sum_y")?.f64()?.get(0).unwrap_or(0.0);
        let sum_xy = stats.column("sum_xy")?.f64()?.get(0).unwrap_or(0.0);
        let sum_x2 = stats.column("sum_x2")?.f64()?.get(0).unwrap_or(0.0);
        let sum_y2 = stats.column("sum_y2")?.f64()?.get(0).unwrap_or(0.0);
        let mean_y = stats.column("mean_y")?.f64()?.get(0).unwrap_or(0.0);

        // 计算线性回归参数
        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;

        // 计算R²
        let ss_tot = sum_y2 - n * mean_y * mean_y;
        let predicted_sum = with_index
            .lazy()
            .with_columns([(lit(slope) * col("x") + lit(intercept)).alias("predicted")])
            .select([((col("y") - col("predicted")).pow(lit(2)))
                .sum()
                .alias("ss_res")])
            .collect()?;

        let ss_res = predicted_sum.column("ss_res")?.f64()?.get(0).unwrap_or(0.0);
        let r_squared = if ss_tot > 0.0 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        };

        Ok(TrendResult {
            slope,
            intercept,
            r_squared,
        })
    }

    fn generate_trend_recommendation(&self, slope: f64, parameter: &str) -> String {
        match slope {
            s if s > 0.05 => format!("{}急剧上升，建议立即检查设备状态", parameter),
            s if s > 0.02 => format!("{}持续上升，建议密切监控", parameter),
            s if s < -0.05 => format!("{}急剧下降，可能存在故障", parameter),
            s if s < -0.02 => format!("{}持续下降，建议检查系统", parameter),
            _ => format!("{}趋势变化", parameter),
        }
    }
}

#[derive(Debug, Clone)]
struct TrendResult {
    slope: f64,
    intercept: f64,
    r_squared: f64,
}

// ========== 4. Polars优化的变化率分析器 ==========

/// 使用Polars窗口函数的变化率分析器
pub struct PolarsChangeRateAnalyzer {
    data_manager: Arc<PolarDataManager>,
    config: ChangeRateConfig,
    name: String,
}

#[derive(Debug, Clone)]
pub struct ChangeRateConfig {
    pub window_size: usize,
    pub instant_rate_threshold: f64,  // 瞬时变化率阈值(%)
    pub period_rate_threshold: f64,   // 周期变化率阈值(%)
    pub critical_rate_threshold: f64, // 临界变化率阈值(%)
}

impl PolarsChangeRateAnalyzer {
    pub fn new(
        name: String,
        data_manager: Arc<PolarDataManager>,
        config: ChangeRateConfig,
    ) -> Self {
        Self {
            data_manager,
            config,
            name,
        }
    }

    /// 使用Polars窗口函数计算变化率
    pub fn analyze_change_rate(
        &self,
        device_id: &str,
        parameter: &str,
        current_value: f64,
        timestamp: DateTime<Utc>,
    ) -> PolarsResult<Option<AnalysisResult>> {
        let timeseries = self.data_manager.get_parameter_timeseries(
            device_id,
            parameter,
            self.config.window_size,
        )?;

        if timeseries.height() < 2 {
            return Ok(None);
        }

        // 使用Polars计算各种变化率
        let rates = self.calculate_rates_with_polars(&timeseries, parameter, current_value)?;

        let is_abnormal = rates.instant_rate.abs() > self.config.instant_rate_threshold
            || rates.period_rate.abs() > self.config.period_rate_threshold;

        if !is_abnormal {
            return Ok(None);
        }

        let severity = if rates.instant_rate.abs() > self.config.critical_rate_threshold
            || rates.period_rate.abs() > self.config.critical_rate_threshold
        {
            Severity::Critical
        } else if rates.instant_rate.abs() > self.config.instant_rate_threshold * 1.5 {
            Severity::High
        } else {
            Severity::Medium
        };

        let mut metadata = HashMap::new();
        metadata.insert("analyzer".to_string(), self.name.clone());
        metadata.insert(
            "instant_rate".to_string(),
            format!("{:.2}%", rates.instant_rate),
        );
        metadata.insert(
            "period_rate".to_string(),
            format!("{:.2}%", rates.period_rate),
        );
        metadata.insert(
            "avg_increase_rate".to_string(),
            format!("{:.2}%", rates.avg_increase_rate.unwrap_or(0.0)),
        );
        metadata.insert(
            "avg_decrease_rate".to_string(),
            format!("{:.2}%", rates.avg_decrease_rate.unwrap_or(0.0)),
        );

        Ok(Some(AnalysisResult {
            device_id: device_id.to_string(),
            parameter: parameter.to_string(),
            warning_type: WarningType::RapidChange,
            severity,
            current_value,
            score: rates.instant_rate.abs().max(rates.period_rate.abs()),
            probability: self.calculate_change_probability(rates.instant_rate, rates.period_rate),
            message: format!(
                "{}变化率异常: 瞬时{:.1}%, 周期{:.1}%",
                parameter, rates.instant_rate, rates.period_rate
            ),
            recommendation: self.generate_rate_recommendation(
                rates.instant_rate,
                rates.period_rate,
                parameter,
            ),
            timestamp,
            metadata,
        }))
    }

    fn calculate_rates_with_polars(
        &self,
        df: &DataFrame,
        parameter: &str,
        current_value: f64,
    ) -> PolarsResult<ChangeRatesSummary> {
        // 使用Polars窗口函数计算变化率
        let with_rates = df
            .clone()
            .lazy()
            .sort(["timestamp"], SortMultipleOptions::default())
            .with_columns([
                // 手动计算瞬时变化率 - 当前值与前一个值的百分比变化
                ((col(parameter) - col(parameter).shift(lit(1)))
                    / col(parameter).shift(lit(1)).abs()
                    * lit(100.0))
                .alias("instant_rate"),
                // 计算第一个值用于周期比较
                col(parameter).first().alias("first_value"),
                col(parameter).last().alias("last_value"),
            ])
            .collect()?;

        // 计算各种变化率
        let first_value = with_rates
            .column("first_value")?
            .f64()?
            .get(0)
            .unwrap_or(0.0);
        let last_value = with_rates
            .column("last_value")?
            .f64()?
            .get(0)
            .unwrap_or(0.0);

        // 瞬时变化率 (与最后一个历史值比较)
        let instant_rate = if last_value != 0.0 {
            ((current_value - last_value) / last_value.abs()) * 100.0
        } else {
            0.0
        };

        // 周期变化率 (与第一个值比较) - 固定值对比
        let period_rate = if first_value != 0.0 {
            ((current_value - first_value) / first_value.abs()) * 100.0
        } else {
            0.0
        };

        // 计算平均增长率和减少率
        let rates_stats = with_rates
            .lazy()
            .filter(col("instant_rate").is_not_null())
            .select([
                col("instant_rate")
                    .filter(col("instant_rate").gt(lit(0.0)))
                    .mean()
                    .alias("avg_increase"),
                col("instant_rate")
                    .filter(col("instant_rate").lt(lit(0.0)))
                    .abs()
                    .mean()
                    .alias("avg_decrease"),
            ])
            .collect()?;

        let avg_increase_rate = rates_stats.column("avg_increase")?.f64()?.get(0);
        let avg_decrease_rate = rates_stats.column("avg_decrease")?.f64()?.get(0);

        Ok(ChangeRatesSummary {
            instant_rate,
            period_rate,
            avg_increase_rate: avg_increase_rate.map(|r| r * 100.0),
            avg_decrease_rate: avg_decrease_rate.map(|r| r * 100.0),
        })
    }

    fn calculate_change_probability(&self, instant_rate: f64, period_rate: f64) -> f64 {
        let max_rate = instant_rate.abs().max(period_rate.abs());
        match max_rate {
            r if r > 50.0 => 0.95,
            r if r > 30.0 => 0.80,
            r if r > 20.0 => 0.60,
            r if r > 15.0 => 0.40,
            _ => 0.20,
        }
    }

    fn generate_rate_recommendation(
        &self,
        instant_rate: f64,
        period_rate: f64,
        parameter: &str,
    ) -> String {
        let max_rate = instant_rate.abs().max(period_rate.abs());
        match max_rate {
            r if r > 50.0 => format!("{}发生剧烈变化，立即停机检查！", parameter),
            r if r > 30.0 => format!("{}变化幅度较大，建议立即检查", parameter),
            r if r > 20.0 => format!("{}变化明显，密切监控", parameter),
            _ => format!("{}出现异常变化", parameter),
        }
    }
}

#[derive(Debug, Clone)]
struct ChangeRatesSummary {
    instant_rate: f64,
    period_rate: f64,
    avg_increase_rate: Option<f64>,
    avg_decrease_rate: Option<f64>,
}

// ========== 主协调系统 ==========

/// Polars优化的石油设备预警系统 (简化版 - 4个核心分析器)
pub struct PolarsOilWarningSystem {
    data_manager: Arc<PolarDataManager>,
    threshold_analyzer: PolarsThresholdAnalyzer,
    statistical_analyzer: PolarsStatisticalAnalyzer,
    trend_analyzer: PolarsTrendAnalyzer,
    rate_analyzer: PolarsChangeRateAnalyzer,
    callbacks: Vec<Box<dyn Fn(&AnalysisResult) + Send + Sync>>,
}

impl PolarsOilWarningSystem {
    pub fn new() -> Self {
        let retention_config = DataRetentionConfig {
            max_rows_per_device: 10000,
            max_days: 90,
        };

        let data_manager = Arc::new(PolarDataManager::new(retention_config));

        let mut threshold_analyzer =
            PolarsThresholdAnalyzer::new("Polars阈值分析器".to_string(), data_manager.clone());
        threshold_analyzer.configure_oil_defaults();

        let statistical_config = StatisticalConfig {
            sigma_threshold: 3.0,
            confidence_level: 0.95,
            min_samples: 30,
            window_size: 200,
        };
        let statistical_analyzer = PolarsStatisticalAnalyzer::new(
            "Polars统计分析器".to_string(),
            data_manager.clone(),
            statistical_config,
        );

        let trend_config = TrendConfig {
            window_size: 50,
            min_points: 10,
            slope_threshold: 0.01,
            r_squared_threshold: 0.7,
        };
        let trend_analyzer = PolarsTrendAnalyzer::new(
            "Polars趋势分析器".to_string(),
            data_manager.clone(),
            trend_config,
        );

        let rate_config = ChangeRateConfig {
            window_size: 20,
            instant_rate_threshold: 15.0,
            period_rate_threshold: 25.0,
            critical_rate_threshold: 40.0,
        };
        let rate_analyzer = PolarsChangeRateAnalyzer::new(
            "Polars变化率分析器".to_string(),
            data_manager.clone(),
            rate_config,
        );

        Self {
            data_manager,
            threshold_analyzer,
            statistical_analyzer,
            trend_analyzer,
            rate_analyzer,
            callbacks: Vec::new(),
        }
    }

    /// 批量添加设备数据
    pub fn add_batch_data(&self, device_id: &str, data_points: Vec<DeviceDataPoint>) -> Result<()> {
        self.data_manager
            .add_batch_data(device_id, data_points)
            .map_err(|e| anyhow::anyhow!("Failed to add batch data: {}", e))
    }

    /// 综合分析 (利用Polars并行处理) - 4个核心分析器
    pub fn analyze_comprehensive(
        &self,
        device_id: &str,
        current_data: DeviceDataPoint,
    ) -> Result<Vec<AnalysisResult>> {
        let mut all_results = Vec::new();

        // 添加当前数据点
        self.add_batch_data(device_id, vec![current_data.clone()])?;

        let parameters = ["pressure", "current", "flow_rate", "temperature"];

        // 并行分析各个参数
        let parameter_results: Vec<Vec<AnalysisResult>> = parameters
            .par_iter()
            .map(|&parameter| {
                let mut results = Vec::new();
                let value = match parameter {
                    "pressure" => current_data.pressure,
                    "current" => current_data.current,
                    "flow_rate" => current_data.flow_rate,
                    "temperature" => current_data.temperature,
                    _ => 0.0,
                };

                // 统计异常分析
                if let Ok(Some(result)) = self.statistical_analyzer.analyze_statistical_anomalies(
                    device_id,
                    parameter,
                    value,
                    current_data.timestamp,
                ) {
                    results.push(result);
                }

                // 趋势分析
                if let Ok(Some(result)) = self.trend_analyzer.analyze_trend(
                    device_id,
                    parameter,
                    value,
                    current_data.timestamp,
                ) {
                    results.push(result);
                }

                // 变化率分析
                if let Ok(Some(result)) = self.rate_analyzer.analyze_change_rate(
                    device_id,
                    parameter,
                    value,
                    current_data.timestamp,
                ) {
                    results.push(result);
                }

                results
            })
            .collect();

        // 合并结果
        for param_results in parameter_results {
            all_results.extend(param_results);
        }

        // 批量阈值分析
        if let Ok(threshold_results) = self.threshold_analyzer.batch_analyze(device_id, 1) {
            all_results.extend(threshold_results);
        }

        // 触发回调
        for result in &all_results {
            for callback in &self.callbacks {
                callback(result);
            }
        }

        Ok(all_results)
    }

    /// 注册回调函数
    pub fn register_callback<F>(&mut self, callback: F)
    where
        F: Fn(&AnalysisResult) + Send + Sync + 'static,
    {
        self.callbacks.push(Box::new(callback));
    }

    /// 获取设备健康报告
    pub fn generate_health_report(&self, device_id: &str) -> Result<DeviceHealthReport> {
        let parameters = ["pressure", "current", "flow_rate", "temperature"];
        let mut parameter_stats = HashMap::new();

        for parameter in parameters {
            if let Ok(stats) = self
                .data_manager
                .calculate_statistics(device_id, parameter, 1000)
            {
                parameter_stats.insert(parameter.to_string(), stats);
            }
        }

        Ok(DeviceHealthReport {
            device_id: device_id.to_string(),
            parameter_statistics: parameter_stats,
            generated_at: Utc::now(),
        })
    }

    /// 获取数据管理器 (用于高级查询)
    pub fn get_data_manager(&self) -> Arc<PolarDataManager> {
        self.data_manager.clone()
    }
}

#[derive(Debug, Clone)]
pub struct DeviceHealthReport {
    pub device_id: String,
    pub parameter_statistics: HashMap<String, StatisticalSummary>,
    pub generated_at: DateTime<Utc>,
}

// ========== 使用示例 ==========

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 Polars优化的石油设备预警分析系统 (简化版)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let mut system = PolarsOilWarningSystem::new();

    // 注册预警回调
    system.register_callback(|result| {
        let severity_icon = match result.severity {
            Severity::Info => "ℹ️",
            Severity::Low => "⚠️",
            Severity::Medium => "🔶",
            Severity::High => "🔴",
            Severity::Critical => "🚨",
        };

        println!(
            "{} [{}] {} - {:?}",
            severity_icon, result.device_id, result.parameter, result.warning_type
        );
        println!(
            "   📊 当前值: {:.2}, 分数: {:.2}, 概率: {:.1}%",
            result.current_value,
            result.score,
            result.probability * 100.0
        );
        println!("   💬 {}", result.message);
        println!("   💡 {}", result.recommendation);

        if let Some(analyzer) = result.metadata.get("analyzer") {
            println!("   🔍 {}", analyzer);
        }
        println!();
    });

    // 准备历史数据
    let mut historical_data = Vec::new();
    for i in 0..100 {
        let timestamp = Utc::now() - chrono::Duration::hours(i);
        let data_point = DeviceDataPoint {
            device_id: "PUMP_001".to_string(),
            timestamp,
            pressure: 2.2 + 0.1 * (i as f64 * 0.1).sin(),
            current: 15.0 + 2.0 * (i as f64 * 0.05).cos(),
            flow_rate: 25.0 + 3.0 * (i as f64 * 0.02).sin(),
            temperature: 45.0 + 5.0 * (i as f64 * 0.01).cos(),
            voltage: 380.0,
            sinking_degree: 120.0,
        };
        historical_data.push(data_point);
    }

    // 批量添加历史数据
    println!("📊 添加历史数据 ({}条)...", historical_data.len());
    system.add_batch_data("PUMP_001", historical_data)?;

    // 模拟正常数据分析
    println!("\n✅ 分析正常数据:");
    let normal_data = DeviceDataPoint {
        device_id: "PUMP_001".to_string(),
        timestamp: Utc::now(),
        pressure: 2.25,
        current: 16.5,
        flow_rate: 24.0,
        temperature: 48.0,
        voltage: 380.0,
        sinking_degree: 115.0,
    };

    let normal_results = system.analyze_comprehensive("PUMP_001", normal_data)?;
    if normal_results.is_empty() {
        println!("   🎯 所有参数正常，无预警");
    }

    // 模拟异常数据分析
    println!("\n🚨 分析异常数据:");
    let abnormal_data = DeviceDataPoint {
        device_id: "PUMP_001".to_string(),
        timestamp: Utc::now(),
        pressure: 5.2,     // 异常高压 - 阈值检测
        current: 35.0,     // 异常高电流 - 阈值检测
        flow_rate: 8.0,    // 异常低流量 - 阈值检测 + 变化率检测
        temperature: 95.0, // 异常高温 - 阈值检测 + 统计异常
        voltage: 380.0,
        sinking_degree: 60.0,
    };

    let _abnormal_results = system.analyze_comprehensive("PUMP_001", abnormal_data)?;

    // 生成健康报告
    println!("\n📋 生成设备健康报告:");
    let health_report = system.generate_health_report("PUMP_001")?;
    println!("设备ID: {}", health_report.device_id);
    println!(
        "报告生成时间: {}",
        health_report.generated_at.format("%Y-%m-%d %H:%M:%S")
    );

    for (param, stats) in &health_report.parameter_statistics {
        println!(
            "  📈 {}: 均值={:.2}, 标准差={:.2}, 中位数={:.2}, 样本数={}",
            param, stats.mean, stats.std_dev, stats.median, stats.count
        );
    }

    // 演示高级Polars查询
    println!("\n🔍 演示高级数据查询功能:");
    let data_manager = system.get_data_manager();
    if let Some(lazy_df) = data_manager.get_device_data("PUMP_001") {
        // 计算每小时的平均值
        let hourly_stats = lazy_df
            .with_columns([(col("timestamp")
                .cast(DataType::Datetime(TimeUnit::Microseconds, None))
                .dt()
                .hour())
            .alias("hour")])
            .group_by([col("hour")])
            .agg([
                col("pressure").mean().alias("avg_pressure"),
                col("current").mean().alias("avg_current"),
                col("pressure").count().alias("data_count"),
            ])
            .sort(["hour"], SortMultipleOptions::default())
            .collect();

        if let Ok(stats_df) = hourly_stats {
            println!("   📊 按小时统计:");
            for i in 0..std::cmp::min(5, stats_df.height()) {
                let hour = stats_df
                    .column("hour")
                    .unwrap()
                    .i32()
                    .unwrap()
                    .get(i)
                    .unwrap_or(0);
                let avg_pressure = stats_df
                    .column("avg_pressure")
                    .unwrap()
                    .f64()
                    .unwrap()
                    .get(i)
                    .unwrap_or(0.0);
                let avg_current = stats_df
                    .column("avg_current")
                    .unwrap()
                    .f64()
                    .unwrap()
                    .get(i)
                    .unwrap_or(0.0);
                let count = stats_df
                    .column("data_count")
                    .unwrap()
                    .u32()
                    .unwrap()
                    .get(i)
                    .unwrap_or(0);
                println!(
                    "      {}时: 平均压力={:.2}, 平均电流={:.2}, 数据点={}",
                    hour, avg_pressure, avg_current, count
                );
            }
        }
    }

    println!("\n✅ 系统特性总结:");
    println!("   🎯 固定阈值分析器: 基于安全规范的硬性限制检测");
    println!("   📊 统计异常分析器: Z-Score + IQR双重统计异常检测");
    println!("   📈 趋势分析器: 线性回归检测参数变化趋势");
    println!("   ⚡ 变化率分析器: 瞬时/周期/平均变化率综合分析");

    Ok(())
}
