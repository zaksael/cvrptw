from dataclasses import dataclass


@dataclass(frozen=True, repr=False)
class Customer:
    cust_id: int
    x: int
    y: int
    demand: int
    ready_time: int
    due_date: int
    service_time: int

    def __repr__(self) -> str:
        return (f"Customer: <{self.cust_id:3}, {self.x:2}, {self.y:2}, {self.demand:2}, "
                f"{self.ready_time:3}, {self.due_date:4}, {self.service_time:2}>")
