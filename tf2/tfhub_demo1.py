import tensorflow_hub as hub
import os
import getpass

module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
print("Loading model from {}".format(module_url))
user = getpass.getuser()
password = getpass.getpass("proxy password:")
os.environ["https_proxy"] = f"http://{user}:{password}@10.204.10.2:3128"
embed = hub.Module(module_url)
