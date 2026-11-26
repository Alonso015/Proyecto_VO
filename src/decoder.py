"""Sequence decoder of the security system.

Stores up to len(password) consecutive letters and, when the sequence is
complete, compares it in order with the password:
  * correct order   -> access granted (unlocked)
  * wrong order     -> the attempt is forgotten and it starts again
An unrecognized stroke, a timeout without new letters or a manual reset
also forget the current sequence.
"""

import time


class SequenceDecoder:
    def __init__(self, password, timeout=20.0):
        self.password = list(password)
        self.timeout = timeout          # seconds without a new letter before forgetting
        self.sequence = []
        self.unlocked = False
        self.message = "Write the password"
        self.last_time = time.time()

    def reset(self, reason="Sequence reset"):
        self.sequence = []
        self.message = reason

    def add(self, letter):
        """Add a recognized letter (None if the stroke was not recognized)."""
        if self.unlocked:
            return
        if letter is None:
            self.reset("Unrecognized stroke: sequence reset")
            return

        self.sequence.append(letter)
        self.last_time = time.time()
        self.message = f"Letter {len(self.sequence)}/{len(self.password)}: {letter}"

        if len(self.sequence) == len(self.password):
            if self.sequence == self.password:
                self.unlocked = True
                self.message = "ACCESS GRANTED"
            else:
                self.reset("Wrong password: try again")

    def update(self):
        """Call every frame: forget the sequence after the timeout."""
        if self.sequence and not self.unlocked and time.time() - self.last_time > self.timeout:
            self.reset("Timeout: sequence reset")

    def lock(self):
        self.unlocked = False
        self.reset("Locked: write the password")

    def display(self):
        """Entered letters followed by empty slots, e.g. 'L Z _ _'."""
        slots = self.sequence + ["_"] * (len(self.password) - len(self.sequence))
        return " ".join(slots)
