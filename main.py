# TODO: make oddessey_agent
# TODO: add argparse for everything

import sys
import argparse
import logging

from odyssey.utils.environment import OdysseyEnvironment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("odyssey.log"), logging.StreamHandler()],
)
logger = logging.getLogger("odyssey-main")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Odyssey - Adversarial Training System"
    )
    # odyssey --theseus_model ppo
    # see mukku's arg parser

    return parser.parse_args()


def setup_python_paths(theseus_path, daedalus_path):
    for path in [theseus_path, daedalus_path]:
        if path not in sys.path:
            sys.path.append(path)
            logger.info(f"Added {path} to Python path")


def main():
    args = parse_args()
    setup_python_paths(args.theseus_path, args.daedalus_path)

    try:
        env = OdysseyEnvironment()

        # Set TCP client connection parameters for Labyrinth
        env.tcp_client.host = "127.0.0.1"  # args.labyrinth_host
        env.tcp_client.port = 4200  # args.labyrinth_port

        logger.info(
            f"Connecting to Labyrinth at {args.labyrinth_host}:{args.labyrinth_port}"
        )

        # Initialize the env
        logger.info("Initializing environment...")
        state = env.initialise_environment()
        logger.info(f"Environment initialized, initial state received")

        # Close the environment
        env.close()
        logger.info("Test complete!")

    except Exception as e:
        logger.error(f"Error in testing: {e}", exc_info=True)
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()


# for odyssey_agent.py:#TODO: add metric tracking
# #TODO: create a reward strategy for daedalus
# #TODO: follow agent_theseus.py's design method
# #TODO: PEP8 based comments + typehinting is a must + black to format

# if __name__ == "__main__":
#     da = DaedalusAgent()
#     ta = TheseusAgent()

#     for ep in episodes:
#         map_data = da([hero_tensor]) #TODO: since parth is not done, make dummy class returns 12x12 map with 7 values (0-7)
#         env.initialise_environment(map_data)

#         while !terminated or !wave_clear:
#             theseus crap

#         daedalus_reward (one value) and give it to da
