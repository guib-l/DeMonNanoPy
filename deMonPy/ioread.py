
import numpy as np


class IOread(object):
    """Base helper providing low-level parsing utilities for output files."""

    def __init__(
        self,
    ):
        """Initialize parsing state."""

        self._block = ""
        self._tocken_block = False

        self.counter = 0

    def compute_block(self, line, ctrl_in="", ctrl_out="", nb_line=None, add=1):
        """Track and accumulate a fixed-size text block.

        Args:
            line: Current line being parsed.
            ctrl_in: Marker indicating the beginning of the block.
            ctrl_out: Unused end-of-block marker placeholder.
            nb_line: Number of lines to collect once the block starts.
            add: Unused compatibility argument.
        """

        if self.is_inside(control=ctrl_in, line=line):
            self._tocken_block = True
            self.counter = nb_line

        if self.counter > 0:
            self._block += line
            self.counter -= 1
        else:
            self._tocken_block = False

    def get_float(self, line, index=-1):
        """Extract a float from a split line.

        Args:
            line: Input line.
            index: Token index to convert.

        Returns:
            float: Parsed floating-point value.
        """
        if index is None:
            raise TypeError("index must be an integer")
        return float(line.split()[index])

    def get_int(self, line, index=-1):
        """Extract an integer from a split line.

        Args:
            line: Input line.
            index: Token index to convert.

        Returns:
            int: Parsed integer value.
        """
        if index is None:
            raise TypeError("index must be an integer")
        return int(line.split()[index])

    def get_list(self, line, index_in=None, index_out=None, ctype=np.float64):
        """Extract a typed slice of tokens from a line.

        Args:
            line: Input line.
            index_in: First token index to include.
            index_out: First token index to exclude.
            ctype: Target NumPy-compatible type constructor.

        Returns:
            Any: Converted token slice.
        """
        if index_in is None:
            return ctype(line.split()[:index_out])
        if index_out is None:
            return ctype(line.split()[index_in:])
        return ctype(line.split()[index_in:index_out])

    def get_dict(
        self,
    ):
        """Return a dictionary representation of parsed content.

        Raises:
            NotImplementedError: Always raised in the base class.
        """
        raise NotImplementedError

    def get_string(self, line, index=None):
        """Extract a string token or return the raw line.

        Args:
            line: Input line.
            index: Optional token index.

        Returns:
            str: Extracted token or original line.
        """
        if index is None:
            return line
        return str(line.split()[index])

    def is_inside(self, control="", line=""):
        """Check whether a control substring is present in a line.

        Args:
            control: Substring to search for.
            line: Line to inspect.

        Returns:
            bool: True when the substring is found.
        """
        if control in line:
            return True
        else:
            return False




