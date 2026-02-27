
from typing import Dict, Any, List


class InputBuilder:
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
        self.lines: List[str] = []
        self.current_line: List[str] = []

    def build(self, user_values: Dict[str, Any]) -> str:
        self.lines = []
        self.current_line = []

        parameters = self.schema.get("PARAMETERS", {})
        for name, definition in parameters.items():
            self._process_parameter(name, definition, user_values.get(name, {}))

        self._flush_line()
        return "\n".join(self.lines)

    def _process_parameter(self, name: str, definition: Dict, value: Any):
        
        param_type = definition.get("TYPE")

        fmt     = definition.get("FORMAT", {})
        inline  = fmt.get("INLINE", False)
        newline = fmt.get("NEWLINE", False)

        if param_type == "BLOCK":
            self._handle_block(name, definition, value, newline)
        
        elif param_type == "FLAG":
            if value:  
                self._write(name, inline)
        
        else:
            if value is None:
                value = definition.get("DEFAULT")

            if value is not None:
                self._write(f"{name} {value}", inline)

    def _handle_block(self, name: str, definition: Dict, value: Dict, newline: bool):
        if value is None:
            return

        self._write(name, inline=False)

        arguments = definition.get("ARGUMENTS", {})
        for arg_name, arg_def in arguments.items():
            user_val = value.get(arg_name, arg_def.get("DEFAULT"))
            self._process_parameter(arg_name, arg_def, user_val)

        if newline:
            self._flush_line()

    def _write(self, text: str, inline: bool):
        if inline:
            self.current_line.append(text)
        else:
            self._flush_line()
            self.current_line.append(text)

    def _flush_line(self):
        if self.current_line:
            self.lines.append(" ".join(self.current_line))
            self.current_line = []



