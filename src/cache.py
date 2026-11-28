from collections import OrderedDict

class LRUCache:

    def __init__(self, capacity: int = 4096):
        if (not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0):
            raise ValueError("Cache capacity must be a positive integer.")

        self.capacity = capacity
        self._cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def __len__(self):
        return len(self._cache)

    def get(self, key: str):
        if key not in self._cache:
            self.misses += 1
            return None

        self.hits += 1
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: str, value: float):
        if key in self._cache:
            self._cache[key] = value
            self._cache.move_to_end(key)
            return
        self._cache[key] = value

        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
            self.evictions += 1

    def clear(self):
        self._cache.clear()
        self.reset_stats()

    def reset_stats(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def stats(self):
        requests = self.hits + self.misses
        hit_rate = (self.hits / requests if requests > 0 else 0.0)

        return {
            "capacity": self.capacity,
            "size": len(self._cache),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "lookups": requests,
            "hit_rate": hit_rate
        }