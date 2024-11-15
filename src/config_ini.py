import os
import configparser
from pint import UnitRegistry as u, Quantity as Q
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ConfigReader:
    _config_data = None

    def __init__(
        self,
        config_file=r"C:\Users\ezmesspc\Documents\Python_Github\EZ-Automat-Teilemess-Rundtische\IndexCalibrationV3.ini",
    ):
        self.config_file_path = config_file
        if ConfigReader._config_data is None:
            self.config = configparser.ConfigParser(
                interpolation=configparser.ExtendedInterpolation()
            )
            self.config.read(config_file)
            ConfigReader._config_data = self.config
        else:
            self.config = ConfigReader._config_data
        self.base_units = {
            "angle": "degree",
            "length": "mm",
            "angle_speed": "degree/second",
            "length_speed": "mm/second",
            "angle_acceleration": "degree/second**2",
            "length_acceleration": "mm/second**2",
            "time": "second",
            "frequency": "hertz",
        }

    def get_config(self):
        return self.config

    def replace_timestamp_placeholder(self, value):
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return value.replace("<YYYY-MM-DD_HH_MM_SS>", current_time)

    def convert_to_base_units(self, value_with_unit):
        # Initialize UnitRegistry
        ureg = u()

        # Split the input string into value and unit
        value, unit = value_with_unit.split()

        # Create a Quantity object
        quantity = Q(float(value), unit)

        # Specific check for frequency units
        if unit in ["Hz", "hertz", "kHz", "MHz", "GHz"]:
            base_quantity = quantity.to("hertz")
            return base_quantity.magnitude, "hertz"

        # Determine the base unit type
        for base_unit_type, base_unit in self.base_units.items():
            try:
                # Convert to base unit directly
                base_quantity = quantity.to(base_unit)

                return base_quantity.magnitude, str(base_quantity.units)
            except:
                pass

        # If no matching base unit type is found, raise an error
        logger.error(f"Unit {unit} is not recognized or not supported.")
        raise ValueError(f"Unit {unit} is not recognized or not supported.")

    def get_converted_value(self, section, option):
        value_with_unit = self.config.get(section, option)
        logger.debug(
            f"Wert mit Einheit - Ausgelesen aus Configfile: {self.config_file_path}: Section: {section}, Option: {option}, Wert: {value_with_unit}"
        )
        magnitude, unit = self.convert_to_base_units(value_with_unit)
        logger.debug(
            f"Ursprünglicher Wert für {option}: {value_with_unit} Konvertierter Wert: {magnitude} {unit}"
        )
        return magnitude

    def get_converted_value_with_unit(self, section, option):
        value_with_unit = self.config.get(section, option)
        logger.debug(
            f"Wert mit Einheit - Ausgelesen aus Configfile: {self.config_file_path}: Section: {section}, Option: {option}, Wert: {value_with_unit}"
        )
        magnitude, unit = self.convert_to_base_units(value_with_unit)
        logger.debug(
            f"Ursprünglicher Wert für {option}: {value_with_unit} Konvertierter Wert: {magnitude} {unit}"
        )
        return magnitude, unit

    def get_value(self, section, option):
        value = self.config.get(section, option)
        logger.debug(
            f"Parameter - Ausgelesen aus Configfile: {self.config_file_path}: Section: {section}, Option: {option}, Parameter: {value}"
        )
        return self.replace_caret_with_double_asterisk(value)

    def replace_caret_with_double_asterisk(self, value):
        return value.replace("^", "**")

    def get_bool(self, section, option):
        value = self.config.get(section, option)
        logger.debug(
            f"Parameter - Ausgelesen aus Configfile: {self.config_file_path}: Section: {section}, Option: {option}, Parameter: {value}"
        )
        return value.lower() == "true"

    def get_path(self, section, option):
        path = self.config.get(section, option)
        logger.debug(
            f"Parameter - Ausgelesen aus Configfile: {self.config_file_path}: Section: {section}, Option: {option}, Pfad: {path}"
        )
        return os.path.expandvars(path)

    def get_file_name(self, section, option):
        name = self.config.get(section, option)
        name = self.replace_timestamp_placeholder(name)
        return name
