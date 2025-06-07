// Import and re-export the Distribution trait
use ndarray::ArrayView1;

pub trait Distribution: Sync + Send {
    fn nll(&self, data: &ArrayView1<f64>) -> f64;
}

// Declare submodules
mod normal;
mod skew_normal;
mod python_callback;

// Re-export all types to maintain the same import structure
pub use normal::{NormalMuNormal, NormalMuNormalSpec};
pub use skew_normal::{NormalMeanPseudoAlphaSkewNormal, NormalMeanPseudoAlphaSkewNormalSpec, NormalMeanNormalGammaSkewNormal, NormalMeanNormalGammaSkewNormalSpec};
pub use python_callback::PythonDistributionWrapper;
