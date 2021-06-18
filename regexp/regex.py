import re

# print(re.match('www', 'www.runoob.com'))  # 在起始位置匹配
# print(re.match('com', 'www.runoob.com'))
#line = "Cats are smarter than dogs"

#matchObj = re.match(r'(.*) are (.*?) .*', line, re.M | re.I)

a="The name \"Hello\" is said \"Halo\" in Spanish."
print(ord('a'))
print(re.match(r'\bThe',a))
print(re.search(r'Halo.*?',a))
print(re.search(r"Halo.*",a))
print(re.search(r"Hello.+",a))