"""Hardware defaults of the onboard Pi with the Rev. G3 tacho shield.

Motor n (1-based) uses DSHOT n on the Pi and TEL n / TACHO n on the Pico board.
Quadcopter: motors 1-4, hexacopter: motors 1-6.
"""

# DSHOT1..6 = GPIO18, 19, 20, 21, 16, 26 (header pins 12, 35, 38, 40, 36, 37)
DSHOT_PINS = [18, 19, 20, 21, 16, 26]
MOTOR_COUNTS = (4, 6)
DEFAULT_MOTORS = 4

PICO_PORT = '/dev/ttyAMA5'
PICO_BAUDRATE = 921600
TRIGGER_PIN = 17

# ESC telemetry on Pi UARTs, only for boards before Rev. G3
LEGACY_ESC_PORTS = ['/dev/ttyAMA0', '/dev/ttyAMA4', '/dev/ttyAMA2', '/dev/ttyAMA3']


def dshot_pins(n_motors=DEFAULT_MOTORS):
    if not 1 <= n_motors <= len(DSHOT_PINS):
        raise ValueError(f'n_motors must be 1..{len(DSHOT_PINS)}')
    return DSHOT_PINS[:n_motors]
