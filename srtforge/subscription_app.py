"""Lightweight monthly subscription manager with a Typer CLI."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
import calendar
from pathlib import Path
from typing import Dict, List, Optional

import typer

DEFAULT_DATA_PATH = Path.home() / ".srtforge_subscriptions.json"


@dataclass
class Plan:
    plan_id: str
    name: str
    price: float
    currency: str = "USD"
    interval_months: int = 1


@dataclass
class Customer:
    customer_id: str
    name: str
    email: str


@dataclass
class Subscription:
    subscription_id: str
    customer_id: str
    plan_id: str
    start_date: date
    next_billing_date: date
    status: str = "active"


class SubscriptionStore:
    """JSON-backed persistence for subscription entities."""

    def __init__(self, path: Path = DEFAULT_DATA_PATH):
        self.path = path
        self._state = self._load()

    @staticmethod
    def add_months(source: date, months: int = 1) -> date:
        month_index = source.month - 1 + months
        year = source.year + month_index // 12
        month = month_index % 12 + 1
        day = min(source.day, calendar.monthrange(year, month)[1])
        return date(year, month, day)

    def _load(self) -> Dict[str, List[Dict]]:
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            return {
                "plans": [self._plan_from_dict(item) for item in raw.get("plans", [])],
                "customers": [self._customer_from_dict(item) for item in raw.get("customers", [])],
                "subscriptions": [
                    self._subscription_from_dict(item) for item in raw.get("subscriptions", [])
                ],
            }
        return {"plans": [], "customers": [], "subscriptions": []}

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "plans": [asdict(plan) for plan in self._state["plans"]],
            "customers": [asdict(customer) for customer in self._state["customers"]],
            "subscriptions": [
                {
                    **asdict(subscription),
                    "start_date": subscription.start_date.isoformat(),
                    "next_billing_date": subscription.next_billing_date.isoformat(),
                }
                for subscription in self._state["subscriptions"]
            ],
        }
        self.path.write_text(json.dumps(payload, indent=2))

    @staticmethod
    def _plan_from_dict(data: Dict) -> Plan:
        return Plan(**data)

    @staticmethod
    def _customer_from_dict(data: Dict) -> Customer:
        return Customer(**data)

    @staticmethod
    def _subscription_from_dict(data: Dict) -> Subscription:
        return Subscription(
            subscription_id=data["subscription_id"],
            customer_id=data["customer_id"],
            plan_id=data["plan_id"],
            start_date=date.fromisoformat(data["start_date"]),
            next_billing_date=date.fromisoformat(data["next_billing_date"]),
            status=data.get("status", "active"),
        )

    def add_plan(self, plan: Plan) -> None:
        if any(existing.plan_id == plan.plan_id for existing in self._state["plans"]):
            raise ValueError(f"Plan with id {plan.plan_id} already exists")
        self._state["plans"].append(plan)
        self._persist()

    def add_customer(self, customer: Customer) -> None:
        if any(existing.customer_id == customer.customer_id for existing in self._state["customers"]):
            raise ValueError(f"Customer with id {customer.customer_id} already exists")
        self._state["customers"].append(customer)
        self._persist()

    def add_subscription(self, subscription: Subscription) -> None:
        if any(existing.subscription_id == subscription.subscription_id for existing in self._state["subscriptions"]):
            raise ValueError(f"Subscription with id {subscription.subscription_id} already exists")
        if not any(plan.plan_id == subscription.plan_id for plan in self._state["plans"]):
            raise ValueError(f"Plan {subscription.plan_id} does not exist")
        if not any(customer.customer_id == subscription.customer_id for customer in self._state["customers"]):
            raise ValueError(f"Customer {subscription.customer_id} does not exist")
        self._state["subscriptions"].append(subscription)
        self._persist()

    def list_plans(self) -> List[Plan]:
        return list(self._state["plans"])

    def list_customers(self) -> List[Customer]:
        return list(self._state["customers"])

    def list_subscriptions(self) -> List[Subscription]:
        return list(self._state["subscriptions"])

    def cancel_subscription(self, subscription_id: str) -> None:
        for subscription in self._state["subscriptions"]:
            if subscription.subscription_id == subscription_id:
                subscription.status = "cancelled"
                self._persist()
                return
        raise ValueError(f"Subscription {subscription_id} not found")

    def bill(self, on_date: date) -> List[Dict[str, str]]:
        invoices: List[Dict[str, str]] = []
        for subscription in self._state["subscriptions"]:
            if subscription.status != "active":
                continue
            if subscription.next_billing_date <= on_date:
                plan = self._find_plan(subscription.plan_id)
                customer = self._find_customer(subscription.customer_id)
                invoices.append(
                    {
                        "subscription_id": subscription.subscription_id,
                        "customer": customer.name,
                        "plan": plan.name,
                        "amount": f"{plan.price:.2f} {plan.currency}",
                        "billing_date": subscription.next_billing_date.isoformat(),
                    }
                )
                subscription.next_billing_date = self.add_months(
                    subscription.next_billing_date, plan.interval_months
                )
        if invoices:
            self._persist()
        return invoices

    def _find_plan(self, plan_id: str) -> Plan:
        for plan in self._state["plans"]:
            if plan.plan_id == plan_id:
                return plan
        raise ValueError(f"Plan {plan_id} not found")

    def _find_customer(self, customer_id: str) -> Customer:
        for customer in self._state["customers"]:
            if customer.customer_id == customer_id:
                return customer
        raise ValueError(f"Customer {customer_id} not found")


def _parse_date(date_str: Optional[str]) -> date:
    return date.fromisoformat(date_str) if date_str else date.today()


def _get_store(path: Optional[Path]) -> SubscriptionStore:
    return SubscriptionStore(path or DEFAULT_DATA_PATH)


app = typer.Typer(add_completion=False, help="Minimal monthly subscription manager")


@app.command()
def create_plan(
    plan_id: str = typer.Argument(..., help="Unique identifier for the plan"),
    name: str = typer.Argument(..., help="Human-friendly plan name"),
    price: float = typer.Argument(..., help="Price per billing interval"),
    currency: str = typer.Option("USD", help="Currency code"),
    interval_months: int = typer.Option(1, help="Months between invoices"),
    db_path: Optional[Path] = typer.Option(None, "--db-path", path_type=Path, help="Storage file"),
) -> None:
    """Create a recurring billing plan."""

    store = _get_store(db_path)
    store.add_plan(Plan(plan_id=plan_id, name=name, price=price, currency=currency, interval_months=interval_months))
    typer.echo(f"Created plan {plan_id} ({name})")


@app.command()
def create_customer(
    customer_id: str = typer.Argument(..., help="Unique identifier for the customer"),
    name: str = typer.Argument(..., help="Customer name"),
    email: str = typer.Argument(..., help="Customer email"),
    db_path: Optional[Path] = typer.Option(None, "--db-path", path_type=Path, help="Storage file"),
) -> None:
    """Create a customer record."""

    store = _get_store(db_path)
    store.add_customer(Customer(customer_id=customer_id, name=name, email=email))
    typer.echo(f"Created customer {customer_id} ({name})")


@app.command()
def subscribe(
    subscription_id: str = typer.Argument(..., help="Unique identifier for the subscription"),
    customer_id: str = typer.Argument(..., help="Existing customer id"),
    plan_id: str = typer.Argument(..., help="Existing plan id"),
    start: Optional[str] = typer.Option(None, "--start", help="Start date (YYYY-MM-DD), defaults to today"),
    db_path: Optional[Path] = typer.Option(None, "--db-path", path_type=Path, help="Storage file"),
) -> None:
    """Subscribe a customer to a plan."""

    store = _get_store(db_path)
    start_date = _parse_date(start)
    next_billing = SubscriptionStore.add_months(start_date)
    store.add_subscription(
        Subscription(
            subscription_id=subscription_id,
            customer_id=customer_id,
            plan_id=plan_id,
            start_date=start_date,
            next_billing_date=next_billing,
        )
    )
    typer.echo(f"Created subscription {subscription_id} for customer {customer_id}")


@app.command()
def bill(
    on: Optional[str] = typer.Option(None, "--on", help="Billing date (YYYY-MM-DD), defaults to today"),
    db_path: Optional[Path] = typer.Option(None, "--db-path", path_type=Path, help="Storage file"),
) -> None:
    """Generate invoices for subscriptions due on or before the selected date."""

    store = _get_store(db_path)
    billing_date = _parse_date(on)
    invoices = store.bill(billing_date)
    if not invoices:
        typer.echo("No invoices due")
        return
    typer.echo(json.dumps(invoices, indent=2))


@app.command("list-plans")
def list_plans(db_path: Optional[Path] = typer.Option(None, "--db-path", path_type=Path, help="Storage file")) -> None:
    """Display all plans."""

    store = _get_store(db_path)
    payload = [asdict(plan) for plan in store.list_plans()]
    typer.echo(json.dumps(payload, indent=2))


@app.command("list-customers")
def list_customers(db_path: Optional[Path] = typer.Option(None, "--db-path", path_type=Path, help="Storage file")) -> None:
    """Display all customers."""

    store = _get_store(db_path)
    payload = [asdict(customer) for customer in store.list_customers()]
    typer.echo(json.dumps(payload, indent=2))


@app.command("list-subscriptions")
def list_subscriptions(db_path: Optional[Path] = typer.Option(None, "--db-path", path_type=Path, help="Storage file")) -> None:
    """Display all subscriptions."""

    store = _get_store(db_path)
    payload = [
        {
            **asdict(subscription),
            "start_date": subscription.start_date.isoformat(),
            "next_billing_date": subscription.next_billing_date.isoformat(),
        }
        for subscription in store.list_subscriptions()
    ]
    typer.echo(json.dumps(payload, indent=2))


if __name__ == "__main__":
    app()
