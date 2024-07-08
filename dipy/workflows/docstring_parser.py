"""
This was taken directly from the file docscrape.py of numpydoc package.

Copyright (C) 2008 Stefan van der Walt <stefan@mentat.za.net>,
Pauli Virtanen <pav@iki.fi>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import re
import textwrap
from warnings import warn

from dipy.testing.decorators import warning_for_keywords


class Reader:
    """A line-based string reader."""

    def __init__(self, data):
        """
        Parameters
        ----------
        data : str
           String with lines separated by '\n'.

        """
        if isinstance(data, list):
            self._str = data
        else:
            self._str = data.split("\n")  # store string as list of lines

        self.reset()

    def __getitem__(self, n):
        return self._str[n]

    def reset(self):
        self._l = 0  # current line nr

    def read(self):
        if not self.eof():
            out = self[self._l]
            self._l += 1
            return out
        else:
            return ""

    def seek_next_non_empty_line(self):
        for ell in self[self._l :]:
            if ell.strip():
                break
            else:
                self._l += 1

    def eof(self):
        return self._l >= len(self._str)

    def read_to_condition(self, condition_func):
        start = self._l
        for line in self[start:]:
            if condition_func(line):
                return self[start : self._l]
            self._l += 1
            if self.eof():
                return self[start : self._l + 1]
        return []

    def read_to_next_empty_line(self):
        self.seek_next_non_empty_line()

        def is_empty(line):
            return not line.strip()

        return self.read_to_condition(is_empty)

    def read_to_next_unindented_line(self):
        def is_unindented(line):
            return line.strip() and (len(line.lstrip()) == len(line))

        return self.read_to_condition(is_unindented)

    def peek(self, n=0):
        if self._l + n < len(self._str):
            return self[self._l + n]
        else:
            return ""

    def is_empty(self):
        return not "".join(self._str).strip()


def dedent_lines(lines):
    """Deindent a list of lines maximally"""
    return textwrap.dedent("\n".join(lines)).split("\n")


class NumpyDocString:
    @warning_for_keywords()
    def __init__(self, docstring, *, config=None):
        docstring = textwrap.dedent(docstring).split("\n")

        self._doc = Reader(docstring)
        self._parsed_data = {
            "Signature": "",
            "Summary": [""],
            "Extended Summary": [],
            "Parameters": [],
            "Outputs": [],
            "Returns": [],
            "Yields": [],
            "Raises": [],
            "Warns": [],
            "Other Parameters": [],
            "Attributes": [],
            "Methods": [],
            "See Also": [],
            "Notes": [],
            "Warnings": [],
            "References": "",
            "Examples": "",
            "index": {},
        }

        self._parse()

    def __getitem__(self, key):
        return self._parsed_data[key]

    def __setitem__(self, key, val):
        if key not in self._parsed_data:
            warn(f"Unknown section {key}", stacklevel=2)
        else:
            self._parsed_data[key] = val

    def _is_at_section(self):
        self._doc.seek_next_non_empty_line()

        if self._doc.eof():
            return False

        l1 = self._doc.peek().strip()  # e.g. Parameters

        if l1.startswith(".. index::"):
            return True

        l2 = self._doc.peek(1).strip()  # ---------- or ==========
        return l2.startswith("-" * len(l1)) or l2.startswith("=" * len(l1))

    def _strip(self, doc):
        _i = 0
        _j = 0
        for _i, line in enumerate(doc):
            if line.strip():
                break

        for _j, line in enumerate(doc[::-1]):
            if line.strip():
                break

        return doc[_i : len(doc) - _j]

    def _read_to_next_section(self):
        section = self._doc.read_to_next_empty_line()

        while not self._is_at_section() and not self._doc.eof():
            if not self._doc.peek(-1).strip():  # previous line was empty
                section += [""]

            section += self._doc.read_to_next_empty_line()

        return section

    def _read_sections(self):
        while not self._doc.eof():
            data = self._read_to_next_section()
            name = data[0].strip()

            if name.startswith(".."):  # index section
                yield name, data[1:]
            elif len(data) < 2:
                yield StopIteration
            else:
                yield name, self._strip(data[2:])

    def _parse_param_list(self, content):
        r = Reader(content)
        params = []
        while not r.eof():
            header = r.read().strip()
            if " : " in header:
                arg_name, arg_type = header.split(" : ")[:2]
            else:
                arg_name, arg_type = header, ""

            desc = r.read_to_next_unindented_line()
            desc = dedent_lines(desc)

            params.append((arg_name, arg_type, desc))

        return params

    _name_rgx = re.compile(
        r"^\s*(:(?P<role>\w+):`(?P<name>[a-zA-Z0-9_.-]+)`|"
        r" (?P<name2>[a-zA-Z0-9_.-]+))\s*",
        re.X,
    )

    def _parse_see_also(self, content):
        """
        func_name : Descriptive text
            continued text
        another_func_name : Descriptive text
        func_name1, func_name2, :meth:`func_name`, func_name3

        """
        items = []

        def parse_item_name(text):
            """Match ':role:`name`' or 'name'"""
            m = self._name_rgx.match(text)
            if m:
                g = m.groups()
                if g[1] is None:
                    return g[3], None
                else:
                    return g[2], g[1]
            raise ValueError(f"{text} is not a item name")

        def push_item(name, rest):
            if not name:
                return
            name, role = parse_item_name(name)
            items.append((name, list(rest), role))
            del rest[:]

        current_func = None
        rest = []

        for line in content:
            if not line.strip():
                continue

            m = self._name_rgx.match(line)
            if m and line[m.end() :].strip().startswith(":"):
                push_item(current_func, rest)
                current_func, line = line[: m.end()], line[m.end() :]
                rest = [line.split(":", 1)[1].strip()]
                if not rest[0]:
                    rest = []
            elif not line.startswith(" "):
                push_item(current_func, rest)
                current_func = None
                if "," in line:
                    for func in line.split(","):
                        if func.strip():
                            push_item(func, [])
                elif line.strip():
                    current_func = line
            elif current_func is not None:
                rest.append(line.strip())
        push_item(current_func, rest)
        return items

    def _parse_index(self, section, content):
        """
        .. index: default
           :refguide: something, else, and more

        """

        def strip_each_in(lst):
            return [s.strip() for s in lst]

        out = {}
        section = section.split("::")
        if len(section) > 1:
            out["default"] = strip_each_in(section[1].split(","))[0]
        for line in content:
            line = line.split(":")
            if len(line) > 2:
                out[line[1]] = strip_each_in(line[2].split(","))
        return out

    def _parse_summary(self):
        """Grab signature (if given) and summary"""
        if self._is_at_section():
            return

        # If several signatures present, take the last one
        while True:
            summary = self._doc.read_to_next_empty_line()
            summary_str = " ".join([s.strip() for s in summary]).strip()
            if re.compile(r"^([\w., ]+=)?\s*[\w\.]+\(.*\)$").match(summary_str):
                self["Signature"] = summary_str
                if not self._is_at_section():
                    continue
            break

        if summary is not None:
            self["Summary"] = summary

        if not self._is_at_section():
            self["Extended Summary"] = self._read_to_next_section()

    def _parse(self):
        self._doc.reset()
        self._parse_summary()

        for section, content in self._read_sections():
            if not section.startswith(".."):
                section = " ".join([s.capitalize() for s in section.split(" ")])
            if section in (
                "Parameters",
                "Outputs",
                "Returns",
                "Raises",
                "Warns",
                "Other Parameters",
                "Attributes",
                "Methods",
            ):
                self[section] = self._parse_param_list(content)
            elif section.startswith(".. index::"):
                self["index"] = self._parse_index(section, content)
            elif section == "See Also":
                self["See Also"] = self._parse_see_also(content)
            else:
                self[section] = content

    # string conversion routines

    def _str_header(self, name, symbol="-"):
        return [name, len(name) * symbol]

    def _str_indent(self, doc, indent=4):
        out = []
        for line in doc:
            out += [" " * indent + line]
        return out

    def _str_signature(self):
        if self["Signature"]:
            return [self["Signature"].replace("*", r"\*")] + [""]
        else:
            return [""]

    def _str_summary(self):
        if self["Summary"]:
            return self["Summary"] + [""]
        else:
            return []

    def _str_extended_summary(self):
        if self["Extended Summary"]:
            return self["Extended Summary"] + [""]
        else:
            return []

    def _str_param_list(self, name):
        out = []
        if self[name]:
            out += self._str_header(name)
            for param, param_type, desc in self[name]:
                if param_type:
                    out += [f"{param} : {param_type}"]
                else:
                    out += [param]
                out += self._str_indent(desc)
            out += [""]
        return out

    def _str_section(self, name):
        out = []
        if self[name]:
            out += self._str_header(name)
            out += self[name]
            out += [""]
        return out

    def _str_see_also(self, func_role):
        if not self["See Also"]:
            return []
        out = []
        out += self._str_header("See Also")
        last_had_desc = True
        for func, desc, role in self["See Also"]:
            if role:
                link = f":{role}:`{func}`"
            elif func_role:
                link = f":{func_role}:`{func}`"
            else:
                link = f"`{func}`_"
            if desc or last_had_desc:
                out += [""]
                out += [link]
            else:
                out[-1] += f", {link}"
            if desc:
                out += self._str_indent([" ".join(desc)])
                last_had_desc = True
            else:
                last_had_desc = False
        out += [""]
        return out

    def _str_index(self):
        idx = self["index"]
        out = []
        out += [f".. index:: {idx.get('default', '')}"]
        for section, references in idx.items():
            if section == "default":
                continue
            out += [f"   :{section}: {', '.join(references)}"]
        return out

    def __str__(self, func_role=""):
        out = []
        out += self._str_signature()
        out += self._str_summary()
        out += self._str_extended_summary()
        for param_list in (
            "Parameters",
            "Returns",
            "Other Parameters",
            "Raises",
            "Warns",
        ):
            out += self._str_param_list(param_list)
        out += self._str_section("Warnings")
        out += self._str_see_also(func_role)
        for s in ("Notes", "References", "Examples"):
            out += self._str_section(s)
        for param_list in ("Attributes", "Methods"):
            out += self._str_param_list(param_list)
        out += self._str_index()
        return "\n".join(out)
