from cvrptw.model import Customer
from cvrptw.operators import segments_cross


def _c(cust_id, x, y):
    return Customer(cust_id=cust_id, x=x, y=y, demand=0, ready_time=0, due_date=9999, service_time=0)


def test_segments_cross():
    # diagonals of a 10×10 square — proper crossing
    assert segments_cross(_c(0, 0, 10), _c(1, 10, 0), _c(2, 0, 0), _c(3, 10, 10))
    # parallel horizontal segments — no crossing
    assert not segments_cross(_c(0, 0, 0), _c(1, 1, 0), _c(2, 0, 1), _c(3, 1, 1))
    # T-intersection: one endpoint on the other segment, no cross-through
    assert not segments_cross(_c(0, 0, 0), _c(1, 2, 0), _c(2, 1, 0), _c(3, 1, 1))
    # shared endpoint
    assert not segments_cross(_c(0, 0, 0), _c(1, 1, 0), _c(2, 1, 0), _c(3, 2, 0))
