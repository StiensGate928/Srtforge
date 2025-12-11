from datetime import date
from pathlib import Path

from srtforge.subscription_app import Customer, Plan, Subscription, SubscriptionStore


def test_add_months_handles_end_of_month():
    start = date(2024, 1, 31)
    result = SubscriptionStore.add_months(start)
    assert result == date(2024, 2, 29)
    second = SubscriptionStore.add_months(result)
    assert second == date(2024, 3, 29)


def test_store_persists_entities(tmp_path: Path):
    path = tmp_path / "subscriptions.json"
    store = SubscriptionStore(path)
    store.add_plan(Plan(plan_id="pro", name="Pro", price=20.0))
    store.add_customer(Customer(customer_id="cust-1", name="Ada", email="ada@example.com"))
    start_date = date(2024, 4, 15)
    subscription = Subscription(
        subscription_id="sub-1",
        customer_id="cust-1",
        plan_id="pro",
        start_date=start_date,
        next_billing_date=SubscriptionStore.add_months(start_date),
    )
    store.add_subscription(subscription)

    reloaded = SubscriptionStore(path)
    assert len(reloaded.list_plans()) == 1
    assert len(reloaded.list_customers()) == 1
    assert len(reloaded.list_subscriptions()) == 1
    saved = reloaded.list_subscriptions()[0]
    assert saved.next_billing_date == date(2024, 5, 15)


def test_billing_advances_next_due_date(tmp_path: Path):
    path = tmp_path / "subscriptions.json"
    store = SubscriptionStore(path)
    store.add_plan(Plan(plan_id="basic", name="Basic", price=10.0, currency="USD"))
    store.add_customer(Customer(customer_id="cust-1", name="Max", email="max@example.com"))
    start_date = date(2024, 2, 1)
    store.add_subscription(
        Subscription(
            subscription_id="sub-2",
            customer_id="cust-1",
            plan_id="basic",
            start_date=start_date,
            next_billing_date=SubscriptionStore.add_months(start_date),
        )
    )

    invoices = store.bill(date(2024, 3, 2))
    assert invoices == [
        {
            "subscription_id": "sub-2",
            "customer": "Max",
            "plan": "Basic",
            "amount": "10.00 USD",
            "billing_date": "2024-03-01",
        }
    ]

    updated = SubscriptionStore(path).list_subscriptions()[0]
    assert updated.next_billing_date == date(2024, 4, 1)
