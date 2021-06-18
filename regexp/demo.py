import re

a="The name \"Hello\" is said \"Halo\" in Spanish."
print(re.match(b"*Halo.*?",a))
print(re.findall(b"+Halo.*",a))
print(re.findall(b"*Hello.+",a))