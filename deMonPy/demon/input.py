




class write_input:

    def __init__(self):
        ...

    def handler_writen(self, params, bind_str=" "):
        """Convert a parameter dictionary into inline deMonNano tokens.

        Args:
            params: Mapping of parameter names to values.

        Returns:
            list[str]: Tokens ready to be written in the input file.
        """

        _inline = []

        for key, item in params.items():
            if isinstance(item, dict):
                _inline.append(self.handler_writen(item))
            elif item is None:
                continue
            elif isinstance(item, bool):
                if item:
                    _inline.append(str(key))
            elif isinstance(item, (float, int)):
                _inline.append(f"{key}={item}")
            elif isinstance(item, str):
                _inline.append(f"{key}{bind_str}{item}")
            else:
                pass
        return _inline

    def _write_table(self, table, symbols=None, fmt="10.8f"):
        """Serialize a generic table of parameters.

        Args:
            table: Mapping of table names to parameter lists.

        Returns:
            list[str]: Serialized table lines.
        """
        out = ""
        if symbols is None:
            symbols = [""] * len(table)
        for symb, values in zip(symbols, table):
            out += f"\n{symb}"
            for v in values:
                out += f" {v:{fmt}}"
        return out



