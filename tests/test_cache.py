import pytest

from src.cache import LRUCache


def test_cache_initialization():

    cache = LRUCache(capacity=3)

    assert len(cache) == 0

    assert cache.capacity == 3


@pytest.mark.parametrize( "capacity", [0, -1, -100, 1.5, True])
def test_invalid_capacity(capacity):
    with pytest.raises(ValueError):
        LRUCache(capacity=capacity)


def test_insert_and_retrieve():

    cache = LRUCache(capacity=3)

    cache.put("ATGCCA", 0.87)

    assert cache.get("ATGCCA") == 0.87

    assert len(cache) == 1


def test_cache_miss():

    cache = LRUCache(capacity=3)

    assert cache.get("UNKNOWN") is None

    assert cache.misses == 1


def test_cache_zero_value():

    cache = LRUCache(capacity=3)

    cache.put("A", 0.0)

    assert cache.get("A") == 0.0

    assert cache.hits == 1


def test_update_existing_key():

    cache = LRUCache(capacity=3)

    cache.put("A", 0.5)

    cache.put("A", 0.8)

    assert cache.get("A") == 0.8

    assert len(cache) == 1


def test_lru_eviction():

    cache = LRUCache(capacity=2)

    cache.put("A", 0.1)
    cache.put("B", 0.2)
    cache.put("C", 0.3)

    assert cache.get("A") is None

    assert cache.get("B") == 0.2
    assert cache.get("C") == 0.3

    assert len(cache) == 2

    assert cache.evictions == 1


def test_access_updates_recency():

    cache = LRUCache(capacity=2)

    cache.put("A", 0.1)
    cache.put("B", 0.2)

    cache.get("A")

    cache.put("C", 0.3)

    assert cache.get("B") is None

    assert cache.get("A") == 0.1
    assert cache.get("C") == 0.3


def test_update_refreshes_recency():

    cache = LRUCache(capacity=2)

    cache.put("A", 0.1)
    cache.put("B", 0.2)

    cache.put("A", 0.9)

    cache.put("C", 0.3)

    assert cache.get("B") is None

    assert cache.get("A") == 0.9


def test_cache_statistics():

    cache = LRUCache(capacity=2)

    cache.put("A", 0.1)

    cache.get("A")
    cache.get("A")

    cache.get("B")

    stats = cache.stats()

    assert stats["hits"] == 2

    assert stats["misses"] == 1

    assert stats["lookups"] == 3

    assert stats["hit_rate"] == pytest.approx(
        2 / 3
    )


def test_empty_cache_statistics():

    cache = LRUCache()

    assert cache.stats()["hit_rate"] == 0.0


def test_reset_stats():

    cache = LRUCache()

    cache.put("A", 0.1)

    cache.get("A")

    cache.reset_stats()

    assert cache.hits == 0

    assert cache.misses == 0

    assert cache.evictions == 0

    assert len(cache) == 1


def test_clear():

    cache = LRUCache()

    cache.put("A", 0.1)

    cache.get("A")

    cache.clear()

    assert len(cache) == 0

    assert cache.hits == 0

    assert cache.misses == 0


def test_capacity_never_exceeded():

    cache = LRUCache(capacity=10)

    for i in range(1000):

        cache.put(str(i), 0.5)

        assert len(cache) <= 10

    assert cache.evictions == 990