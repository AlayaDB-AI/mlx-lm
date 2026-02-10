import multiprocessing

from alayajet.api_server.server import main


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
