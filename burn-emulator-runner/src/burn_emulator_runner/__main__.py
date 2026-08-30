import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "burn_emulator_runner.app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8080")),
        workers=1,
    )


if __name__ == "__main__":
    main()
