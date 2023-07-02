class StackIndexesGenerator:
    def __init__(self, size: int, step: int, position: str = "last"):
        self.size = size
        self.step = step

        if position == "first":
            self.behind = 0
            self.ahead = self.size - 1
        elif position == "middle":
            self.behind = self.size // 2
            self.ahead = self.size - self.behind - 1
        elif position == "last":
            self.behind = self.size - 1
            self.ahead = 0
        else:
            raise ValueError(
                f"Index position value should be one of {'first', 'middle', 'last'}"
            )
        self.behind *= self.step
        self.ahead *= self.step

    def make_stack_indexes(self, frame_index: int) -> list[int]:
        return list(
            range(
                frame_index - self.behind,
                frame_index + self.ahead + 1,
                self.step,
            )
        )

    def clip_index(self, index: int, frame_count: int, save_zone: int = 0) -> int:
        behind_frames = self.behind + save_zone
        ahead_frames = self.ahead + save_zone
        if index < behind_frames:
            index = behind_frames
        elif index >= frame_count - ahead_frames:
            index = frame_count - ahead_frames - 1
        return index
