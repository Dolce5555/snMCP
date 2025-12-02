from ruamel.yaml import YAML

filePath = "/app/backend/secrets.yaml"
yaml = YAML(typ = "safe", pure = True)
with open(filePath, "r", encoding="utf-8") as f:
    secrets = yaml.load(f)

print(secrets)
print("\\\n---\\\n")
print(yaml.dump(secrets))