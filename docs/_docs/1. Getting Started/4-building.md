---
title: "Building"
permalink: /building/
excerpt: "Instructions on building."
last_modified_at: 2021-01-16T12:30:00-00:00
toc: false
classes: wide
---

This project is firmly requiring the use of Python PIP with [PEP 517][pep517] support. This means you need pip >= [19.0][pip19].

Considering version [19.0][pip19] released on the 22nd of January 2019, it isn't much of an ask in my opinion, when you end up
with an overall much smoother build experience.

## Building distribution files

    pip install build
    git clone https://github.com/rlaPHOENiX/vsgan && cd vsgan
    python -m build

To install the built distribution files, install the .whl file available in /dist, e.g. `pip install dist/*.whl`

## Installing from source

    git clone https://github.com/rlaPHOENiX/vsgan && cd vsgan
    pip install .

[pep517]: https://www.python.org/dev/peps/pep-0517

[pip19]: https://pip.pypa.io/en/stable/news/#id415
