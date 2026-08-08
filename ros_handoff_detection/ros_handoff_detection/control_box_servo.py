import time
from typing import Optional

import serial


class ServoController:
    def __init__(
        self,
        port: str,
        baud_rate: int = 9600,
        startup_delay: float = 2.0,
    ) -> None:
        self.port = port
        self.baud_rate = baud_rate
        self.startup_delay = startup_delay
        self._serial: Optional[serial.Serial] = None

    def connect(self) -> None:
        """Open the connection to the Arduino."""
        if self._serial is not None and self._serial.is_open:
            return

        self._serial = serial.Serial(
            port=self.port,
            baudrate=self.baud_rate,
            timeout=1,
        )

        # Most Arduino boards reset when the serial port opens.
        time.sleep(self.startup_delay)

    def send_command(self, command: int) -> None:
        """
        Send a command to the Arduino.

        0: move the servo to 0 degrees
        1: move the servo to 90 degrees
        """
        if command not in (0, 1):
            raise ValueError("Command must be either 0 or 1.")

        if self._serial is None or not self._serial.is_open:
            raise RuntimeError("Arduino is not connected. Call connect() first.")

        self._serial.write(str(command).encode("ascii"))
        self._serial.flush()

    def close(self) -> None:
        """Close the Arduino connection."""
        if self._serial is not None and self._serial.is_open:
            self._serial.close()

    def __enter__(self) -> "ServoController":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def main() -> None:
    with ServoController("COM3") as servo:
        servo.send_command(1)  # Move to 90 degrees
        time.sleep(2)

        servo.send_command(0)  # Move to 0 degrees


if __name__ == "__main__":
    main()