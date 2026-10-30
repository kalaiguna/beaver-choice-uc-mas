import pandas as pd
import numpy as np
import os
import time
import difflib
import re
import dotenv
import ast
from dataclasses import dataclass, field
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from sqlalchemy import create_engine, Engine
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11-100 units: 1 day
        - 101-1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]


########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################

# ---------------------------------------------------------------------------
# Architecture overview (see workflow_diagram.png and the report)
#
#   Orchestrator agent (customer-facing)
#     |-- Inventory agent : stock checks, supplier restocking, post-sale reorder review
#     |-- Quoting agent   : quote history lookup, bulk-discount pricing
#     |-- Sales agent     : order verification and finalisation (database writes)
#
# Business rules (stock, deadlines, cash reserve, discount tiers) live in plain
# Python tools so they are applied consistently. The language models handle
# understanding the customer's wording and deciding which tool to call next.
# Data moves between agents through a shared, typed RequestState object.
# ---------------------------------------------------------------------------

# ----- Configuration --------------------------------------------------------

dotenv.load_dotenv()

OPENAI_BASE_URL = "https://openai.vocareum.com/v1"
MODEL_NAME = "gpt-4o-mini"  # keep to the models named in the course to protect the budget
MODEL_SETTINGS = ModelSettings(temperature=0)

# Internal business parameters (never shown to customers).
CASH_RESERVE = 5000.0        # cash that must remain after any supplier purchase
SUPPLIER_COST_RATIO = 0.70   # supplier price as a fraction of the list price
DEFAULT_MIN_STOCK = 100      # reorder threshold for items without a stored minimum
REAM_SIZE = 500              # sheets in one ream
BULK_DISCOUNT_TIERS = [      # (minimum total units, discount rate)
    (5000, 0.15),
    (1000, 0.10),
    (500, 0.05),
]
MAX_UNITS_PER_LINE = 20000    # larger lines are declined as too big for a single standard order
MAX_EXTRA_HISTORY_DISCOUNT = 0.03  # extra discount the quoting agent may add, based on past quotes

CATALOG: Dict[str, dict] = {item["item_name"]: item for item in paper_supplies}

# The catalog only stocks A4 among the ISO A sizes, so A3, A5 and so on are different products.
UNSUPPORTED_SIZE_PATTERN = re.compile(r"\ba([0-35-9])\b", re.IGNORECASE)


def build_model() -> OpenAIChatModel:
    """Create the chat model that every agent uses, routed through the Vocareum proxy."""
    api_key = os.getenv("UDACITY_OPENAI_API_KEY")
    if not api_key or api_key.startswith("voc-..."):
        raise RuntimeError("Set UDACITY_OPENAI_API_KEY in the .env file before running the agents.")
    return OpenAIChatModel(
        MODEL_NAME,
        provider=OpenAIProvider(base_url=OPENAI_BASE_URL, api_key=api_key),
    )


# ----- Typed data shared between agents ------------------------------------

class OrderLine(BaseModel):
    """One product the customer asked for."""
    customer_description: str = Field(description="The product exactly as the customer described it")
    quantity: int = Field(description="The number exactly as the customer wrote it. Never convert or multiply it.", gt=0)
    quantity_in_reams: bool = Field(
        default=False, description="True only if the customer literally wrote 'ream' or 'reams' for this line")


class ParsedOrder(BaseModel):
    """A customer request converted into structured data."""
    lines: List[OrderLine]
    deadline_date: Optional[str] = Field(
        default=None, description="Date the customer needs the delivery by, as YYYY-MM-DD, if stated"
    )


class CustomerReply(BaseModel):
    """The orchestrator's final output. The order details are added by code, not by the model."""
    opening: str = Field(description="One friendly sentence acknowledging the request. No numbers, prices or promises.")
    closing: str = Field(description="One short polite closing sentence. No numbers, prices or promises.")


@dataclass
class LineState:
    """Everything the agents learn about one requested line."""
    line_index: int
    customer_description: str
    quantity: int
    item_name: Optional[str] = None      # exact catalog name, None if we do not sell it
    unit_price: float = 0.0
    stock_on_request_date: int = 0
    recommended_restock: int = 0
    ready_date: Optional[str] = None     # date the units are on hand
    restocked: bool = False
    available: bool = False
    reason: str = ""                     # customer-safe explanation when not available
    line_total: float = 0.0              # price after discount, filled by the quoting agent
    fulfilled: bool = False              # filled by the sales agent


@dataclass
class RequestState:
    """Shared state for one customer request (passed to every agent as dependencies)."""
    request_date: str
    deadline_date: Optional[str] = None
    pending_order: Optional[ParsedOrder] = None
    lines: Dict[int, LineState] = field(default_factory=dict)   # keyed by line_index
    subtotal: float = 0.0
    discount_rate: float = 0.0
    total: float = 0.0
    quote_explanation: str = ""
    discount_text: str = ""
    sale_summary: str = ""
    sale_finalized: bool = False
    inventory_checked: bool = False
    quote_done: bool = False
    trace: List[str] = field(default_factory=list)              # tool calls, for the console log


# ----- Small helpers ---------------------------------------------------------

def log_step(ctx: RunContext[RequestState], text: str) -> None:
    """Record a tool call in the request trace."""
    ctx.deps.trace.append(text)


def to_iso_date(date_text: str) -> str:
    """Normalise a date string to YYYY-MM-DD."""
    return datetime.fromisoformat(date_text.strip()[:10]).strftime("%Y-%m-%d")


def units_in_stock(item_name: str, as_of_date: str) -> int:
    """Net units of an item on hand at a date (wraps get_stock_level)."""
    frame = get_stock_level(item_name, as_of_date)
    return int(frame["current_stock"].iloc[0])


def available_units(item_name: str, as_of_date: str) -> int:
    """Units that can still be promised to a new order.

    Earlier customers' sales can be dated in the future (on their restock delivery date), so stock on the
    request date alone would over-promise. We take the lower of today's stock and the projected stock after
    all known future movements.
    """
    return min(units_in_stock(item_name, as_of_date), units_in_stock(item_name, "2099-12-31"))


def committed_cash(as_of_date: str) -> float:
    """Cash we can safely count on: the lower of the balance at a date and after all known future movements."""
    return min(get_cash_balance(as_of_date), get_cash_balance("2099-12-31"))


def ensure_inventory_record(item_name: str, quantity: int) -> None:
    """Make sure a restocked catalog item is in the inventory table, so reports value it and the review watches it."""
    existing = pd.read_sql("SELECT 1 FROM inventory WHERE item_name = :name", db_engine, params={"name": item_name})
    if existing.empty:
        item = CATALOG[item_name]
        pd.DataFrame([{"item_name": item_name, "category": item["category"], "unit_price": item["unit_price"],
                       "current_stock": quantity, "min_stock_level": DEFAULT_MIN_STOCK}]
                     ).to_sql("inventory", db_engine, if_exists="append", index=False)


def minimum_stock_level(item_name: str) -> int:
    """Reorder threshold for an item, taken from the inventory table when available."""
    frame = pd.read_sql(
        "SELECT min_stock_level FROM inventory WHERE item_name = :name",
        db_engine,
        params={"name": item_name},
    )
    return int(frame["min_stock_level"].iloc[0]) if not frame.empty else DEFAULT_MIN_STOCK


def bulk_discount_rate(total_units: int) -> float:
    """Volume discount for an order, using BULK_DISCOUNT_TIERS."""
    for minimum_units, rate in BULK_DISCOUNT_TIERS:
        if total_units >= minimum_units:
            return rate
    return 0.0


def restock_cost(item_name: str, quantity: int) -> float:
    """What the supplier charges us for a restock order."""
    return round(CATALOG[item_name]["unit_price"] * SUPPLIER_COST_RATIO * quantity, 2)


def is_before_or_on(date_a: str, date_b: Optional[str]) -> bool:
    """True when date_a is on or before date_b (a missing date_b means no limit)."""
    return date_b is None or date_a <= date_b


def find_line_for_item(state: RequestState, item_name: str) -> Optional[LineState]:
    """The line of this request that was matched to a catalog item, preferring one that still needs a restock."""
    candidates = [line for line in state.lines.values() if line.item_name == item_name]
    needing_restock = [line for line in candidates if line.recommended_restock > 0 and not line.restocked]
    return (needing_restock or candidates or [None])[0]


def build_customer_summary(state: RequestState) -> str:
    """Customer-facing summary of what was confirmed and what could not be supplied.

    Built from the recorded state (not by a language model), so amounts and reasons are always accurate
    and contain no internal information.
    """
    ordered_lines = [state.lines[i] for i in sorted(state.lines)]
    shipped = [line for line in ordered_lines if line.fulfilled]
    not_shipped = [line for line in ordered_lines if not line.fulfilled]
    parts = []
    if shipped:
        parts.append("Confirmed items:")
        for line in shipped:
            when = ("in stock, ready to ship now" if not line.restocked
                    else f"restocked, ready to ship on {line.ready_date}")
            parts.append(f"- {line.quantity:,} x {line.item_name}: ${line.line_total:,.2f} ({when})")
        confirmed_total = sum(line.line_total for line in shipped)
        latest_ready = max(line.ready_date for line in shipped)
        parts.append(f"Order total: ${confirmed_total:,.2f}. "
                     f"Prices are our standard per-unit prices with {state.discount_text}.")
        timing = f"All confirmed items are ready to ship by {latest_ready}"
        if state.deadline_date:
            timing += f", in time for your requested date of {state.deadline_date}"
        parts.append(timing + ".")
    if not_shipped:
        parts.append("Items we could not supply:")
        for line in not_shipped:
            reason = line.reason or "We could not guarantee this item with the stock and timing available."
            parts.append(f"- {line.quantity:,} x {line.customer_description}: {reason}")
    return "\n".join(parts)


# ============================================================================
# INVENTORY AGENT TOOLS
# helpers used: get_stock_level, get_supplier_delivery_date, get_cash_balance,
#               create_transaction, get_all_inventory, generate_financial_report
# ============================================================================

def search_catalog(ctx: RunContext[RequestState], description: str) -> List[dict]:
    """Find catalog products that match a customer's wording.

    Args:
        description: Product wording from the customer, for example "glossy A4 paper".

    Returns the closest catalog items with their exact item_name and list unit_price.
    Only use item names returned here. If nothing fits, the product is not sold.
    """
    wanted = description.lower()
    wanted_tokens = set(wanted.replace("-", " ").replace("(", " ").replace(")", " ").split())
    scored = []
    for name, item in CATALOG.items():
        name_tokens = set(name.lower().replace("-", " ").replace("(", " ").replace(")", " ").split())
        overlap = len(wanted_tokens & name_tokens) / max(len(name_tokens), 1)
        similarity = difflib.SequenceMatcher(None, wanted, name.lower()).ratio()
        scored.append((overlap + similarity, name, item))
    scored.sort(reverse=True)
    matches = [{"item_name": name, "category": item["category"], "unit_price": item["unit_price"]}
               for _, name, item in scored[:5]]
    log_step(ctx, f"search_catalog('{description}') -> {matches[0]['item_name']}")
    return matches


def check_stock_for_line(ctx: RunContext[RequestState], line_index: int, item_name: Optional[str]) -> dict:
    """Check whether one requested line can be supplied, and record the result.

    Call this exactly once for every requested line.

    Args:
        line_index: The number of the requested line (as listed in the task, starting at 0).
        item_name: Exact catalog name from search_catalog, or null if we do not sell the product.

    Returns stock on hand, whether a supplier restock could arrive before the customer's
    deadline, and the quantity recommended for restocking. If restock_needed and
    restock_possible are both true, call restock_item.
    """
    state = ctx.deps
    order = state.pending_order
    if order is None or not 0 <= line_index < len(order.lines):
        return {"error": "Unknown line_index."}

    existing = state.lines.get(line_index)
    if existing is not None:  # each line is evaluated once; repeating the call must not repeat purchases
        return {"already_checked": True, "item_name": existing.item_name, "available": existing.available,
                "restock_needed": False, "restock_possible": False, "note": "This line was already checked."}

    requested = order.lines[line_index]
    quantity = requested.quantity
    line = LineState(line_index=line_index, customer_description=requested.customer_description,
                     quantity=quantity, item_name=item_name)
    state.lines[line_index] = line

    size_not_stocked = UNSUPPORTED_SIZE_PATTERN.search(requested.customer_description) is not None
    if item_name is None or item_name not in CATALOG or size_not_stocked:
        line.item_name = None
        line.reason = ("This size is not part of our current range." if size_not_stocked
                       else "This product is not part of our current range.")
        log_step(ctx, f"check_stock_for_line({line_index}) -> not in range")
        return {"found_in_catalog": False, "restock_needed": False, "restock_possible": False,
                "note": line.reason}

    if quantity > MAX_UNITS_PER_LINE:
        line.item_name = None
        line.reason = ("This quantity is larger than we can supply in a single order. "
                       "Please contact us to arrange a custom order.")
        log_step(ctx, f"check_stock_for_line({line_index}) -> quantity {quantity:,} over single-order limit")
        return {"found_in_catalog": True, "restock_needed": False, "restock_possible": False,
                "note": "Quantity is over the single-order limit."}

    if not is_before_or_on(state.request_date, state.deadline_date):
        line.reason = f"The requested date of {state.deadline_date} has already passed."
        log_step(ctx, f"check_stock_for_line({line_index}) -> deadline already passed")
        return {"found_in_catalog": True, "restock_needed": False, "restock_possible": False,
                "note": "The customer's deadline has already passed."}

    line.unit_price = CATALOG[item_name]["unit_price"]
    # Stock already promised to other lines of this same request is not available again.
    # A restocked line only takes the units we already had; the rest comes from its own restock.
    reserved = sum((max(other.stock_on_request_date, 0) if other.restocked else other.quantity)
                   for other in state.lines.values()
                   if other is not line and other.item_name == item_name and other.available)
    line.stock_on_request_date = available_units(item_name, state.request_date) - reserved

    if line.stock_on_request_date >= quantity:
        line.available = True
        line.ready_date = state.request_date
        log_step(ctx, f"check_stock_for_line({line_index}) -> {item_name}: in stock")
        return {"found_in_catalog": True, "item_name": item_name, "stock": line.stock_on_request_date,
                "restock_needed": False, "restock_possible": False, "note": "Enough stock on hand."}

    shortfall = quantity - line.stock_on_request_date
    # Add a small safety buffer only if it does not push the order into a slower delivery tier.
    buffered = shortfall + max(0, minimum_stock_level(item_name) // 2)
    base_eta = get_supplier_delivery_date(state.request_date, shortfall)
    buffered_eta = get_supplier_delivery_date(state.request_date, buffered)
    line.recommended_restock = buffered if buffered_eta == base_eta else shortfall
    eta = get_supplier_delivery_date(state.request_date, line.recommended_restock)

    deadline_ok = is_before_or_on(eta, state.deadline_date)
    cash_after = committed_cash(eta) - restock_cost(item_name, line.recommended_restock)
    affordable = cash_after >= CASH_RESERVE

    if not deadline_ok:
        line.reason = (f"We cannot restock this item in time. The earliest we could have it is {eta}, "
                       f"which is after your requested date of {state.deadline_date}.")
    elif not affordable:
        line.reason = "We are unable to source this quantity at the moment."
    else:
        line.reason = "We could not guarantee this item with the stock and timing available."  # replaced if restock succeeds

    log_step(ctx, f"check_stock_for_line({line_index}) -> {item_name}: short by {shortfall}, "
                  f"restock possible={deadline_ok and affordable}")
    return {"found_in_catalog": True, "item_name": item_name, "stock": line.stock_on_request_date,
            "shortfall": shortfall, "restock_needed": True,
            "restock_possible": deadline_ok and affordable,
            "recommended_restock_quantity": line.recommended_restock, "supplier_delivery_date": eta,
            "meets_customer_deadline": deadline_ok, "within_cash_limits": affordable}


def restock_item(ctx: RunContext[RequestState], item_name: str, quantity: int) -> dict:
    """Order stock from the supplier for an item. Guard rails are enforced here.

    Args:
        item_name: Exact catalog name.
        quantity: Units to order (use recommended_restock_quantity from check_stock_for_line,
            or the suggested reorder quantity from inventory_health_check).

    For a customer order, the quantity must cover the shortfall and the order is refused if it would
    arrive after the customer's deadline. All purchases are refused if they would leave the company
    below its cash reserve. The stock appears in the database on the delivery date.
    """
    state = ctx.deps
    if item_name not in CATALOG or quantity <= 0:
        return {"ordered": False, "note": "Unknown item or invalid quantity."}

    line = find_line_for_item(state, item_name)
    if line is not None:  # restock for a customer order
        if line.restocked:
            return {"ordered": False, "note": "This line was already restocked."}
        if line.recommended_restock <= 0:
            return {"ordered": False, "note": "No restock is needed for this item."}
        shortfall = line.quantity - line.stock_on_request_date
        if quantity < shortfall:
            return {"ordered": False, "note": f"Quantity must be at least {shortfall}."}
        quantity = min(quantity, max(line.recommended_restock, shortfall))

    eta = get_supplier_delivery_date(state.request_date, quantity)
    if line is not None and not is_before_or_on(eta, state.deadline_date):
        return {"ordered": False, "note": f"Delivery {eta} is after the customer deadline."}

    cost = restock_cost(item_name, quantity)
    if committed_cash(eta) - cost < CASH_RESERVE:
        return {"ordered": False, "note": "Not enough cash headroom for this purchase."}

    transaction_id = create_transaction(item_name, "stock_orders", quantity, cost, eta)
    ensure_inventory_record(item_name, quantity)
    if line is not None:
        line.restocked = True
        line.available = True
        line.ready_date = eta
        line.reason = ""
    log_step(ctx, f"restock_item({item_name}, {quantity}) -> arrives {eta}")
    return {"ordered": True, "transaction_id": transaction_id, "quantity": quantity,
            "supplier_delivery_date": eta}


def inventory_health_check(ctx: RunContext[RequestState]) -> dict:
    """Internal review of stock levels and finances (never shared with customers).

    Returns items at or below their minimum stock level, counting deliveries already on order,
    plus the cash balance and inventory valuation. Use it after a sale to decide what to reorder.
    """
    date = ctx.deps.request_date
    # Projected stock: includes supplier deliveries already on order (dated in the future),
    # so the same item is not reordered again and again.
    projected = get_all_inventory("2099-12-31")
    inventory_table = pd.read_sql("SELECT item_name, min_stock_level FROM inventory", db_engine)

    low_stock = []
    for _, row in inventory_table.iterrows():
        name, minimum = row["item_name"], int(row["min_stock_level"])
        stock = int(projected.get(name, 0))
        if stock <= minimum:
            low_stock.append({"item_name": name, "projected_stock": stock, "min_stock_level": minimum,
                              "suggested_reorder_quantity": minimum * 2 - stock})

    report = generate_financial_report(date)
    log_step(ctx, f"inventory_health_check -> {len(low_stock)} item(s) below minimum")
    return {"low_stock_items": low_stock,
            "cash_balance": round(float(report["cash_balance"]), 2),
            "inventory_value": round(float(report["inventory_value"]), 2),
            "total_assets": round(float(report["total_assets"]), 2)}


# ============================================================================
# QUOTING AGENT TOOLS
# helpers used: search_quote_history
# ============================================================================

def lookup_quote_history(ctx: RunContext[RequestState], search_terms: List[str]) -> List[dict]:
    """Look up similar past quotes to learn how discounts were applied.

    Args:
        search_terms: One or two short keywords (for example ["cardstock"]). A quote must
            match ALL terms, so more terms means fewer results.
    """
    history = search_quote_history(search_terms, limit=3)
    log_step(ctx, f"lookup_quote_history({search_terms}) -> {len(history)} past quote(s)")
    return [{"total_amount": quote["total_amount"],
             "order_size": quote["order_size"],
             "quote_explanation": str(quote["quote_explanation"] or "")[:300]} for quote in history]


def calculate_quote(ctx: RunContext[RequestState], extra_history_discount: float = 0.0) -> dict:
    """Price the lines that can be supplied and store the quote.

    Args:
        extra_history_discount: Extra discount rate between 0 and 0.03, only when past quotes
            for similar orders show a clearly higher discount than the standard tier.

    The standard bulk discount is chosen from the total units quoted. Lines that cannot be
    supplied are left out and keep their reason.
    """
    state = ctx.deps
    quotable = [line for line in state.lines.values() if line.available]
    if not quotable:
        state.subtotal = state.discount_rate = state.total = 0.0
        state.quote_explanation = "None of the requested items can be supplied."
        return {"quoted_lines": 0, "total": 0.0}

    total_units = sum(line.quantity for line in quotable)
    extra = min(max(extra_history_discount, 0.0), MAX_EXTRA_HISTORY_DISCOUNT)
    volume_rate = bulk_discount_rate(total_units)
    rate = round(volume_rate + extra, 4)

    state.subtotal = round(sum(line.quantity * line.unit_price for line in quotable), 2)
    for line in quotable:
        line.line_total = round(line.quantity * line.unit_price * (1 - rate), 2)
    state.discount_rate = rate
    state.total = round(sum(line.line_total for line in quotable), 2)

    reasons = []
    if volume_rate > 0:
        reasons.append(f"a {volume_rate:.0%} volume discount for ordering {total_units:,} units in total")
    if extra > 0:
        reasons.append(f"an extra {extra:.0%} discount in line with pricing on similar past orders")
    discount_text = " and ".join(reasons) if reasons else (
        "no discount, because the order is below our volume discount thresholds")
    state.discount_text = discount_text
    state.quote_explanation = (f"Prices are our standard per-unit prices with {discount_text}. "
                               f"Subtotal ${state.subtotal:,.2f}, total ${state.total:,.2f}.")
    log_step(ctx, f"calculate_quote -> {len(quotable)} line(s), discount {rate:.0%}, total ${state.total:,.2f}")
    return {"quoted_lines": len(quotable), "total_units": total_units, "subtotal": state.subtotal,
            "discount_rate": rate, "total": state.total, "explanation": state.quote_explanation}


# ============================================================================
# SALES AGENT TOOLS
# helpers used: get_stock_level, create_transaction
# ============================================================================

def line_can_ship(state: RequestState, line: LineState) -> bool:
    """A line ships only if it is available, priced, in stock on its ready date and on time."""
    return (line.available and line.item_name is not None and line.line_total > 0
            and units_in_stock(line.item_name, line.ready_date) >= line.quantity
            and is_before_or_on(line.ready_date, state.deadline_date))


def verify_order_feasibility(ctx: RunContext[RequestState]) -> List[dict]:
    """Check each quoted line against stock on its ready date and the customer's deadline.

    Returns one entry per line with can_ship true or false. Read-only.
    """
    state = ctx.deps
    checks = [{"line_index": line.line_index, "item_name": line.item_name, "quantity": line.quantity,
               "ready_date": line.ready_date, "can_ship": line_can_ship(state, line)}
              for line in sorted(state.lines.values(), key=lambda l: l.line_index)]
    log_step(ctx, f"verify_order_feasibility -> {sum(c['can_ship'] for c in checks)} of {len(checks)} can ship")
    return checks


def finalize_order(ctx: RunContext[RequestState]) -> str:
    """Record the sale for every line that can ship and return the customer summary.

    This writes 'sales' transactions to the database. It only ever runs once per request.
    """
    state = ctx.deps
    if state.sale_finalized:
        return state.sale_summary

    for line in sorted(state.lines.values(), key=lambda l: l.line_index):
        if line_can_ship(state, line):
            create_transaction(line.item_name, "sales", line.quantity, line.line_total, line.ready_date)
            line.fulfilled = True
    state.sale_finalized = True
    state.sale_summary = build_customer_summary(state)
    log_step(ctx, f"finalize_order -> {sum(l.fulfilled for l in state.lines.values())} line(s) sold")
    return state.sale_summary


# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

def build_agents(model: Model):
    """Create the four agents. Returns (orchestrator, inventory, quoting, sales)."""

    inventory_agent = Agent(
        model, deps_type=RequestState, model_settings=MODEL_SETTINGS, retries=2,
        tools=[search_catalog, check_stock_for_line, restock_item, inventory_health_check],
        instructions=(
            "You are the inventory agent of Beaver's Choice Paper Company. "
            "For an order: for EVERY requested line, call search_catalog with the customer's wording to find "
            "the exact catalog item, then call check_stock_for_line with the line's index and the exact item_name "
            "(use null if nothing reasonable matches). Different sizes (A3, A5 ...) and different products are never substitutes for each other: only match the same product. When the customer gives dimensions (for example 24x36 inches), prefer the catalog item whose name states those dimensions. "
            "If it says restock_needed and restock_possible, call restock_item with "
            "recommended_restock_quantity. Never restock when restock_possible is false. "
            "For a post-sale review: call inventory_health_check, then call restock_item for each low stock item "
            "using its suggested_reorder_quantity. "
            "Reply with a short plain summary of what you did."
        ),
    )

    quoting_agent = Agent(
        model, deps_type=RequestState, model_settings=MODEL_SETTINGS, retries=2,
        tools=[lookup_quote_history, calculate_quote],
        instructions=(
            "You are the quoting agent of Beaver's Choice Paper Company. "
            "Call lookup_quote_history once with one or two keywords for the main product to see how similar "
            "past orders were discounted. Then call calculate_quote. Only pass extra_history_discount "
            "(max 0.03) if the past quotes clearly show a higher discount than a normal volume discount; "
            "otherwise leave it at 0. Reply with one sentence about how the quote was priced."
        ),
    )

    sales_agent = Agent(
        model, deps_type=RequestState, model_settings=MODEL_SETTINGS, retries=2,
        tools=[verify_order_feasibility, finalize_order],
        instructions=(
            "You are the sales agent of Beaver's Choice Paper Company. "
            "Call verify_order_feasibility first, then finalize_order exactly once. "
            "Reply with the exact text returned by finalize_order."
        ),
    )

    orchestrator = Agent(
        model, deps_type=RequestState, output_type=CustomerReply,
        model_settings=MODEL_SETTINGS, retries=2,
        instructions=(
            "You are the customer-facing orchestrator for Beaver's Choice Paper Company. "
            "Handle each request in this order:\n"
            "1. Call register_order with the structured order (copy quantities exactly as written and set quantity_in_reams only if the customer literally says reams; deadline as YYYY-MM-DD).\n"
            "2. Call delegate_to_inventory_agent and wait for its result. Call one tool at a time, never several in the same step.\n"
            "3. If at least one item can be supplied, call delegate_to_quoting_agent, and wait for its result, and only then delegate_to_sales_agent. "
            "If nothing can be supplied, skip those two.\n"
            "4. Call delegate_to_inventory_agent_review once at the end.\n"
            "5. Finish with a short opening sentence and closing sentence. The order details, prices, discounts, "
            "timing and reasons are added automatically, so do not repeat or invent any numbers, dates, prices "
            "or reasons, and never mention internal costs, margins, cash, tools or errors."
        ),
    )

    # ----- Orchestrator tools: typed hand-offs to the worker agents -----
    @orchestrator.tool
    def register_order(ctx: RunContext[RequestState], order: ParsedOrder) -> str:
        """Store the structured order. Call this first.

        Args:
            order: The requested lines and the delivery deadline.
        """
        try:
            ctx.deps.deadline_date = to_iso_date(order.deadline_date) if order.deadline_date else None
        except ValueError:
            raise ModelRetry("deadline_date must be formatted YYYY-MM-DD, or null if the customer gave no date.")
        for line in order.lines:  # reams are converted here, in code, never by the model
            if line.quantity_in_reams:
                line.quantity *= REAM_SIZE
                line.quantity_in_reams = False
        ctx.deps.pending_order = order
        log_step(ctx, f"register_order -> {len(order.lines)} line(s), deadline {ctx.deps.deadline_date}")
        return f"Order registered with {len(order.lines)} lines."

    @orchestrator.tool
    async def delegate_to_inventory_agent(ctx: RunContext[RequestState]) -> str:
        """Ask the inventory agent to check stock and restock for the registered order."""
        state = ctx.deps
        order = state.pending_order
        if order is None:
            return "No order registered yet. Call register_order first."

        def task_text(indexes):
            lines = "\n".join(f"- line {i}: {order.lines[i].quantity} x {order.lines[i].customer_description}"
                              for i in indexes)
            return (f"Request date: {state.request_date}. Customer deadline: {state.deadline_date}.\n"
                    f"Requested lines:\n{lines}")

        await inventory_agent.run(task_text(range(len(order.lines))), deps=state, usage=ctx.usage)
        missing = [i for i in range(len(order.lines)) if i not in state.lines]
        if missing:  # coverage check: every requested line must have been looked at
            await inventory_agent.run(task_text(missing), deps=state, usage=ctx.usage)
            for i in range(len(order.lines)):
                if i not in state.lines:
                    state.lines[i] = LineState(
                        line_index=i, customer_description=order.lines[i].customer_description,
                        quantity=order.lines[i].quantity,
                        reason="We could not guarantee this item with the stock and timing available.")
        state.inventory_checked = True
        supplied = [line for line in state.lines.values() if line.available]
        return f"{len(supplied)} of {len(state.lines)} lines can be supplied."

    @orchestrator.tool
    async def delegate_to_quoting_agent(ctx: RunContext[RequestState]) -> str:
        """Ask the quoting agent to price the lines that can be supplied."""
        if not ctx.deps.inventory_checked:
            return "Not yet: call delegate_to_inventory_agent first and wait for its result."
        await quoting_agent.run("Quote the order.", deps=ctx.deps, usage=ctx.usage)
        ctx.deps.quote_done = True
        return ctx.deps.quote_explanation

    @orchestrator.tool
    async def delegate_to_sales_agent(ctx: RunContext[RequestState]) -> str:
        """Ask the sales agent to finalise the sale."""
        if not ctx.deps.quote_done:
            return "Not yet: call delegate_to_quoting_agent first and wait for its result."
        await sales_agent.run("Finalise the order.", deps=ctx.deps, usage=ctx.usage)
        return "Sale finalised." if ctx.deps.sale_finalized else "The sale could not be finalised."

    @orchestrator.tool
    async def delegate_to_inventory_agent_review(ctx: RunContext[RequestState]) -> str:
        """Ask the inventory agent for a post-sale stock review (internal, nothing to tell the customer)."""
        # The review gets its own state so it can never alter the customer's order.
        review_state = RequestState(request_date=ctx.deps.request_date)
        await inventory_agent.run(
            f"Post-sale review as of {review_state.request_date}: reorder low stock items.",
            deps=review_state, usage=ctx.usage)
        ctx.deps.trace.extend(f"[review] {step}" for step in review_state.trace)
        return "Review complete."

    return orchestrator, inventory_agent, quoting_agent, sales_agent


# ============================================================================
# RUNNING A REQUEST
# ============================================================================

DEFAULT_OPENING = "Thank you for your request to Beaver's Choice Paper Company."
DEFAULT_CLOSING = "Please let us know if we can help with anything else."


def safe_sentence(text: str, fallback: str) -> str:
    """Use the model's opening or closing sentence only if it makes no numeric promises."""
    cleaned = text.strip()
    if not cleaned or re.search(r"[0-9$]", cleaned):
        return fallback
    return cleaned


def handle_customer_request(orchestrator: Agent, request_text: str, request_date: str) -> tuple:
    """Run one customer request through the multi-agent system.

    Returns (customer_response, RequestState). The response is the orchestrator's opening and closing
    sentences around a summary that is generated from the recorded state.
    """
    state = RequestState(request_date=request_date)
    try:
        result = orchestrator.run_sync(request_text, deps=state,
                                       usage_limits=UsageLimits(request_limit=60))
        reply = result.output
        summary = state.sale_summary or build_customer_summary(state)
        if not summary:
            summary = "We were unable to supply any of the requested items."
        return (f"{safe_sentence(reply.opening, DEFAULT_OPENING)}\n\n{summary}\n\n{safe_sentence(reply.closing, DEFAULT_CLOSING)}"), state
    except Exception as error:  # never show internal errors to the customer
        print(f"INTERNAL ERROR while handling request: {error}")
        if any(line.fulfilled for line in state.lines.values()):
            # The sale is already recorded, so tell the customer what was confirmed.
            summary = state.sale_summary or build_customer_summary(state)
            return f"{DEFAULT_OPENING}\n\n{summary}\n\n{DEFAULT_CLOSING}", state
        return ("We are sorry, we could not process your request right now. "
                "Please try again shortly."), state


def request_status(state: RequestState) -> str:
    """Label for test_results.csv: fulfilled, partially fulfilled or not fulfilled."""
    fulfilled = sum(1 for line in state.lines.values() if line.fulfilled)
    if state.lines and fulfilled == len(state.lines):
        return "fulfilled"
    return "partially fulfilled" if fulfilled else "not fulfilled"


# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():

    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    # Initialise the multi-agent system
    orchestrator, *_ = build_agents(build_model())

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        response, state = handle_customer_request(orchestrator, request_with_date, request_date)

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print("Agent actions:")
        for step in state.trace:
            print(f"  - {step}")
        print(f"Response: {response}")
        print(f"Status: {request_status(state)}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "status": request_status(state),
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
