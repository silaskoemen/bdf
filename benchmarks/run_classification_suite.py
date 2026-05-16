"""Classification benchmark suite entry point.

Runs CustomOrchestrator for the classification task. Target-type filtering
is handled inside the orchestrator via the model's `target_type` config
field.

Usage (from repo root):
    pixi run clas-suite model=bdf_betamvbernoulli
"""

from loguru import logger

from .pipeline.orchestrators import CustomOrchestrator
from .utils.benchmark_utils import setup_logging

if __name__ == "__main__":
    import hydra
    from omegaconf import OmegaConf

    @hydra.main(config_path="./configs/", config_name="config", version_base="1.3")
    def main(cfg: OmegaConf) -> None:
        setup_logging(cfg.logging)
        logger.info("🚀 Starting classification benchmark suite")
        logger.debug(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
        orchestrator = CustomOrchestrator(cfg)
        results = orchestrator.run()
        logger.success("🏁 Classification benchmark suite completed")
        logger.debug(f"Final results:\n{results}")

    main()
