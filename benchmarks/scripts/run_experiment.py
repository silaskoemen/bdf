from orchestrators import Orchestrator

if __name__ == "__main__":
    import hydra
    from omegaconf import OmegaConf

    @hydra.main(config_path="../configs/", config_name="config")
    def main(cfg: OmegaConf) -> None:
        print(cfg)
        orchestrator = Orchestrator(cfg)
        orchestrator.run()

    main()
