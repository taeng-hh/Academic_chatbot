import os


def pytest_addoption(parser):
    parser.addoption('--run-live-api', action='store_true', default=False,
                     help='Enable paid smoke tests against the configured OpenAI API')


def pytest_configure(config):
    if config.getoption('--run-live-api'):
        os.environ['RUN_LIVE_API'] = '1'
