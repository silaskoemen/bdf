use std::time::Instant;

// ============================================================================
// CONFIGURATION
// ============================================================================

const N_SAMPLES: usize = 1_000_000;   // 1M rows
const ETA: f64 = 0.01;              // Check 100 thresholds (1/0.01)
const SPEEDUP_FACTOR: u64 = 10;     // Fast score is 10x faster (conservative for O(N) vs O(1))
const BASE_WORK_ITER: u64 = 500;    // "Work" units for the fast function

// ============================================================================
// MOCK INFRASTRUCTURE
// ============================================================================

#[derive(Clone, Copy, Default)]
struct SufficientStats {
    n: f64,
    sum: f64,
    sum_sq: f64,
}

impl SufficientStats {
    fn add(&mut self, val: f64) {
        self.n += 1.0;
        self.sum += val;
        self.sum_sq += val * val;
    }
    fn remove(&mut self, val: f64) {
        self.n -= 1.0;
        self.sum -= val;
        self.sum_sq -= val * val;
    }
}

// Simulates O(1) scoring (Sufficient Stats)
// Just does some math operations
fn mock_fast_score(stats: &SufficientStats) -> f64 {
    let mut res = 0.0;
    // Simulate calculation overhead
    for _ in 0..BASE_WORK_ITER {
        res += stats.sum * 0.0001;
    }
    res
}

// Simulates O(N) scoring (Standard)
// Iterates over the data slice AND has higher base overhead
fn mock_slow_score(data: &[f64]) -> f64 {
    let mut res = 0.0;

    // 1. The cost of iterating the data (O(N))
    for &x in data {
        res += x;
    }

    // 2. The cost of the math (slower than fast score)
    for _ in 0..(BASE_WORK_ITER * SPEEDUP_FACTOR) {
        res += 0.0001;
    }
    res
}

fn main() {
    println!("Generating data (N={}, ETA={})...", N_SAMPLES, ETA);

    // Generate random data
    let x: Vec<f64> = (0..N_SAMPLES).map(|i| (i as f64 * 0.5).sin()).collect();
    let y: Vec<f64> = (0..N_SAMPLES).map(|i| (i as f64 * 0.1).cos()).collect();

    // ========================================================================
    // APPROACH 1: STANDARD (Quantile + Allocation + Slow Score)
    // ========================================================================
    let start_std = Instant::now();

    // 1. Sort feature to find quantiles
    let mut x_sorted = x.clone();
    x_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n_thresholds = (1.0 / ETA).ceil() as usize;
    let stride = N_SAMPLES / n_thresholds;

    let mut thresholds = Vec::with_capacity(n_thresholds);
    for i in 1..n_thresholds {
        if i * stride < N_SAMPLES {
            thresholds.push(x_sorted[i * stride]);
        }
    }

    let mut std_checksum = 0.0;

    // 2. Iterate thresholds
    for &thresh in &thresholds {
        // 3. ALLOCATION COST: Filter data into new arrays (simulating Array1 creation)
        // This is the hidden killer in the current implementation
        let left_y: Vec<f64> = y.iter()
            .zip(x.iter())
            .filter(|(_, &val)| val <= thresh)
            .map(|(&y_val, _)| y_val)
            .collect();

        let right_y: Vec<f64> = y.iter()
            .zip(x.iter())
            .filter(|(_, &val)| val > thresh)
            .map(|(&y_val, _)| y_val)
            .collect();

        // 4. SCORING COST: O(N)
        std_checksum += mock_slow_score(&left_y);
        std_checksum += mock_slow_score(&right_y);
    }

    let duration_std = start_std.elapsed();
    println!("Standard Approach: {:.2?}", duration_std);

    // ========================================================================
    // APPROACH 2: SORT-AND-SCAN (Sufficient Stats + O(1) Score)
    // ========================================================================
    let start_scan = Instant::now();

    // 1. Pre-calc parent stats
    let mut parent_stats = SufficientStats::default();
    for &val in y.iter() { parent_stats.add(val); }

    // 2. Sort indices (O(N log N))
    let mut indices: Vec<usize> = (0..N_SAMPLES).collect();
    indices.sort_unstable_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());

    let mut left_stats = SufficientStats::default();
    let mut right_stats = parent_stats;
    let mut scan_checksum = 0.0;

    // 3. Scan (O(N))
    // We iterate every point to update stats, but only score at quantiles
    let scan_stride = (N_SAMPLES as f64 * ETA).max(1.0) as usize;

    for (i, &idx) in indices.iter().enumerate() {
        let val = y[idx];

        // O(1) Update
        left_stats.add(val);
        right_stats.remove(val);

        // Only score at quantiles
        if (i + 1) % scan_stride == 0 {
            // O(1) Score
            scan_checksum += mock_fast_score(&left_stats);
            scan_checksum += mock_fast_score(&right_stats);
        }
    }

    let duration_scan = start_scan.elapsed();
    println!("Sort-and-Scan:     {:.2?}", duration_scan);

    // ========================================================================
    // RESULTS
    // ========================================================================
    let speedup = duration_std.as_secs_f64() / duration_scan.as_secs_f64();
    println!("------------------------------------------------");
    println!("Speedup Factor:    {:.2}x", speedup);
    println!("(Checksums: {:.2} vs {:.2})", std_checksum, scan_checksum);
}
