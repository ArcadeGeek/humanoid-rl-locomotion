from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


DT = 0.02
BRAKE_S = 0.36
BRAKE_STEPS = int(round(BRAKE_S / DT))

# Provisional engineering-only thresholds for FUNCTIONAL router testing.
# They are not scientifically calibrated/frozen by this unit test.
ENTER_LOCO_SPEED = 0.20
EXIT_LOCO_SPEED = 0.10

PASS_MARKER = "V2G_V4N_ROUTER_HYSTERESIS_OFFLINE_UNIT_TEST_PASS"


class State(Enum):
    STAND = auto()
    LOCO = auto()
    BRAKE = auto()


@dataclass(frozen=True)
class Output:
    state: State
    active_policy: str
    policy_command_speed: float
    brake_step: int


class Router:
    def __init__(
        self,
        enter_loco_speed: float,
        exit_loco_speed: float,
        brake_steps: int,
    ) -> None:
        if not (0.0 <= exit_loco_speed < enter_loco_speed):
            raise ValueError("require 0 <= EXIT < ENTER")
        if brake_steps <= 0:
            raise ValueError("brake_steps must be positive")

        self.enter = float(enter_loco_speed)
        self.exit = float(exit_loco_speed)
        self.brake_steps = int(brake_steps)
        self.state = State.STAND
        self._brake_count = 0

    def update(self, requested_speed: float) -> Output:
        speed = abs(float(requested_speed))

        if self.state is State.STAND:
            if speed >= self.enter:
                self.state = State.LOCO

        elif self.state is State.LOCO:
            if speed <= self.exit:
                self.state = State.BRAKE
                self._brake_count = 0

        elif self.state is State.BRAKE:
            # Abort braking if locomotion demand clearly returns.
            if speed >= self.enter:
                self.state = State.LOCO
                self._brake_count = 0
            else:
                self._brake_count += 1
                if self._brake_count >= self.brake_steps:
                    self.state = State.STAND
                    self._brake_count = 0

        else:
            raise RuntimeError(f"unknown state: {self.state}")

        if self.state is State.STAND:
            return Output(
                state=self.state,
                active_policy="v2-G",
                policy_command_speed=0.0,
                brake_step=0,
            )

        if self.state is State.LOCO:
            return Output(
                state=self.state,
                active_policy="v4-N",
                policy_command_speed=speed,
                brake_step=0,
            )

        # BRAKE is always v4-N at zero command.
        return Output(
            state=self.state,
            active_policy="v4-N",
            policy_command_speed=0.0,
            brake_step=self._brake_count,
        )


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    check(BRAKE_STEPS == 18, f"expected 18 brake steps, got {BRAKE_STEPS}")

    r = Router(
        enter_loco_speed=ENTER_LOCO_SPEED,
        exit_loco_speed=EXIT_LOCO_SPEED,
        brake_steps=BRAKE_STEPS,
    )

    print("=" * 116)
    print("v2-G / v4-N ROUTER + HYSTERESIS OFFLINE UNIT TEST")
    print("=" * 116)
    print(f"dt                       : {DT:.3f} s")
    print(f"brake dwell              : {BRAKE_S:.2f} s = {BRAKE_STEPS} steps")
    print(f"ENTER_LOCO_SPEED         : {ENTER_LOCO_SPEED:.3f} m/s  [PROVISIONAL / ENGINEERING-ONLY]")
    print(f"EXIT_LOCO_SPEED          : {EXIT_LOCO_SPEED:.3f} m/s  [PROVISIONAL / ENGINEERING-ONLY]")
    print("policy weights           : ABSENT")
    print("Isaac Sim                : ABSENT")
    print("physics                  : ABSENT")
    print("purpose                  : state-machine/invariant validation only")
    print("=" * 116)

    # 1) STAND must not leave on a command inside hysteresis band.
    out = r.update(0.15)
    check(out.state is State.STAND, "STAND incorrectly left inside hysteresis band")
    check(out.active_policy == "v2-G", "STAND must select v2-G")
    print("[PASS] STAND holds inside hysteresis band")

    # 2) Enter locomotion only at/above ENTER threshold.
    out = r.update(0.25)
    check(out.state is State.LOCO, "failed STAND->LOCO transition")
    check(out.active_policy == "v4-N", "LOCO must select v4-N")
    check(abs(out.policy_command_speed - 0.25) < 1e-12, "LOCO command not forwarded")
    print("[PASS] STAND->LOCO at ENTER threshold regime")

    # 3) Once in LOCO, hysteresis band must retain LOCO.
    out = r.update(0.15)
    check(out.state is State.LOCO, "LOCO incorrectly exited inside hysteresis band")
    print("[PASS] LOCO holds inside hysteresis band")

    # 4) Low command must enter BRAKE, not direct STAND.
    out = r.update(0.05)
    check(out.state is State.BRAKE, "LOCO did not enter BRAKE")
    check(out.active_policy == "v4-N", "BRAKE must retain v4-N")
    check(out.policy_command_speed == 0.0, "BRAKE must override v4-N command to zero")
    print("[PASS] LOCO->BRAKE uses v4-N zero-command conditioning")

    # 5) BRAKE must last exactly 18 router updates after entry.
    # Entry update above initializes count=0; now perform 17 more BRAKE updates.
    for i in range(BRAKE_STEPS - 1):
        out = r.update(0.0)
        check(out.state is State.BRAKE, f"BRAKE ended early at follow-up step {i + 1}")
        check(out.active_policy == "v4-N", "BRAKE switched policy early")
        check(out.policy_command_speed == 0.0, "BRAKE command became nonzero")

    out = r.update(0.0)
    check(out.state is State.STAND, "BRAKE did not complete after 18 follow-up updates")
    check(out.active_policy == "v2-G", "post-BRAKE state must select v2-G")
    print("[PASS] 0.36 s braking dwell completes before v2-G STAND")

    # 6) STAND must still hold in the same hysteresis band.
    out = r.update(0.15)
    check(out.state is State.STAND, "STAND chatter inside hysteresis band after return")
    print("[PASS] post-handoff STAND hysteresis hold")

    # 7) Enter LOCO again.
    out = r.update(0.40)
    check(out.state is State.LOCO, "failed second STAND->LOCO transition")

    # 8) Enter BRAKE, then abort if a clear locomotion command returns.
    out = r.update(0.0)
    check(out.state is State.BRAKE, "second LOCO->BRAKE failed")
    for _ in range(5):
        out = r.update(0.0)
        check(out.state is State.BRAKE, "BRAKE ended unexpectedly before abort test")

    out = r.update(0.40)
    check(out.state is State.LOCO, "BRAKE did not abort on renewed locomotion demand")
    check(out.active_policy == "v4-N", "abort must remain on v4-N")
    check(abs(out.policy_command_speed - 0.40) < 1e-12, "renewed command not forwarded")
    print("[PASS] BRAKE cancellation on renewed locomotion demand")

    # 9) Explicit architecture invariant: no blending surface exists.
    check(
        set(Output.__dataclass_fields__) == {
            "state", "active_policy", "policy_command_speed", "brake_step"
        },
        "unexpected output surface; blending-like field introduced",
    )
    print("[PASS] no blending path")

    print("=" * 116)
    print("ROUTER LOGIC RESULT       : FUNCTIONAL")
    print("THRESHOLD STATUS          : PROVISIONAL / NOT CALIBRATED")
    print("DWELL STATUS              : FROZEN 0.36 s")
    print(PASS_MARKER)
    print("=" * 116)


if __name__ == "__main__":
    main()
