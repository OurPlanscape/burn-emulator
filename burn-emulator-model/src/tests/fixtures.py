from burn_emulator.constants import Path

TEST_BATCH_SIZE = 32
TEST_IMAGE_SIZE = 129
TEST_IN_CHANS = 19
TEST_OUT_CHANS = 3

TEST_FIXTURES_DIR = Path(__file__).parent.parent.parent / "configs"
TEST_CONFIG_PATH = TEST_FIXTURES_DIR / "ignition_dataset_config.yaml"
