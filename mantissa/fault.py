"""What can go wrong: the machine stops rather than guess."""


class Fault(Exception):
    """Something the machine refuses to do.

    kind is "illegal-instruction", "address" or "not-a-character"; detail is a
    readable explanation.
    """

    def __init__(self, kind, detail):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail
