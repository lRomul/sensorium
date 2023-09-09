import queue
from multiprocessing import Queue

from rosny import ProcessStream, ComposeStream

from torch.utils.data._utils.collate import default_collate


class WorkerStream(ProcessStream):
    def __init__(self,
                 dataset,
                 index_queue: Queue,
                 result_queue: Queue,
                 timeout: float = 1.0):
        super().__init__()
        self._dataset = dataset
        self._index_queue = index_queue
        self._result_queue = result_queue
        self._timeout = timeout

    def work(self):
        try:
            index = self._index_queue.get(timeout=self._timeout)
        except queue.Empty:
            return
        sample = self._dataset[index]
        self._result_queue.put(sample)


class ComposeWorkersStream(ComposeStream):
    def __init__(self,
                 num_workers: int,
                 dataset,
                 index_queue: Queue,
                 result_queue: Queue):
        super().__init__()
        for index in range(num_workers):
            worker = WorkerStream(dataset, index_queue, result_queue)
            self.__setattr__(f"worker_{index}", worker)


class DataLoader:
    def __init__(self,
                 dataset,
                 batch_size: int,
                 num_workers: int):
        assert num_workers > 0
        self.dataset = dataset
        self.num_workers = num_workers
        self.batch_size = batch_size

        self._index_queue = Queue(maxsize=len(self.dataset))
        self._result_queue = Queue(maxsize=self.batch_size)

        self._num_samples_left = 0

        self._workers_stream = ComposeWorkersStream(
            self.num_workers, self.dataset, self._index_queue, self._result_queue,
        )
        self.start_workers()

    def start_workers(self):
        self._workers_stream.start()

    def stop_workers(self):
        if not self._workers_stream.stopped():
            self._workers_stream.stop()
        if not self._workers_stream.joined():
            self._workers_stream.join()

    def clear_queues(self):
        while not self._index_queue.empty():
            self._index_queue.get()
        while not self._result_queue.empty():
            self._result_queue.get()

    def __iter__(self):
        self._num_samples_left = len(self.dataset)
        self.clear_queues()
        for index in range(len(self.dataset)):
            self._index_queue.put(index)
        return self

    def __next__(self):
        batch_list = []
        while self._num_samples_left:
            sample = self._result_queue.get()
            batch_list.append(sample)
            self._num_samples_left -= 1
            if len(batch_list) == self.batch_size:
                return default_collate(batch_list)
        if batch_list:
            return default_collate(batch_list)
        self.clear_queues()
        raise StopIteration

    def __del__(self):
        self.stop_workers()
