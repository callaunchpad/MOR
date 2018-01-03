import yaml

class Config():

	def __init__(self, filename = 'Config.yaml'):
		custom_config = yaml.load(open(filename, 'r'))["config"]
		default_config = yaml.load(open('Default.yaml', 'r'))["default_config"]
		self.config = self.resolve_config(default_config, custom_config)

	def resolve_config(self, default_config, custom_config):
		"""
		Given two config dict objects, merge them into a new dict using a shallow copy.
		Args:
		    default_config (dict): Full dictionary with default settings
		    custom_config (dict): Config values loaded from custom yaml file
		Returns:
			config (dict): Full config dictionary with default settings merged and overridden with custom values
		"""
		config = default_config.copy()
		config.update(custom_config)
		return config