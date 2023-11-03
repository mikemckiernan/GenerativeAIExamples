"""The definition of the application configuration."""
from frontend.configuration_wizard import ConfigWizard, configclass, configfield


@configclass
class AppConfig(ConfigWizard):
    """Configuration class for the application.

    :cvar triton: The configuration of the chat server.
    :type triton: ChatConfig
    :cvar model: The configuration of the model
    :type triton: ModelConfig
    """

    server_url: str = configfield(
        "serverUrl",
        default="http://10.110.17.73",
        help_txt="The location of the chat server.",
    )
    server_port: str = configfield(
        "serverPort",
        default="8000",
        help_txt="The port on which the chat server is listening for HTTP requests.",
    )
    model_name: str = configfield(
        "modelName",
        default="llama2-7B-chat",
        help_txt="The name of the hosted LLM model.",
    )
