# Beaver's Choice Paper Company: Multi-Agent Quoting and Inventory System

Reflection report for the multi-agent project. The system is built with **pydantic-ai** and uses four agents, below the limit of five. The implementation is the single file `project_starter.py`. The workflow diagram is `workflow_diagram.png` and the evaluation output is `test_results.csv`.

## 1. System overview

The company receives free-text paper orders. For each order the system must check stock, decide whether to buy from the supplier, price the order with sensible discounts, and finalize the sale, while telling the customer clearly what was confirmed and why anything was declined.

The workflow in `workflow_diagram.png` is:

1. The customer request arrives at the **Orchestrator**.
2. The Orchestrator turns it into a typed order and asks the **Inventory agent** to check stock and restock where needed.
3. If at least one item can be supplied, the **Quoting agent** prices those items and the **Sales agent** finalizes the sale.
4. The Inventory agent runs a post-sale stock review (in its own isolated state) and reorders low-stock items.
5. The customer receives a reply built from the recorded results.

## 2. Agents, responsibilities and tools

| Agent | Responsibility (no overlap with the others) | Tools | Starter helper functions used |
|---|---|---|---|
| **Orchestrator** | Customer-facing. Parses the request into a typed order, delegates in a fixed order, writes only an opening and a closing sentence. | `register_order`, `delegate_to_inventory_agent`, `delegate_to_quoting_agent`, `delegate_to_sales_agent`, `delegate_to_inventory_agent_review` | none directly (delegates only) |
| **Inventory** | Matches customer wording to catalog items, checks stock, decides when to restock from the supplier, and reviews low stock after each sale. | `search_catalog`, `check_stock_for_line`, `restock_item`, `inventory_health_check` | `get_stock_level`, `get_supplier_delivery_date`, `get_cash_balance`, `create_transaction`, `get_all_inventory`, `generate_financial_report` |
| **Quoting** | Looks at similar past quotes and prices the items that can be supplied, with bulk discounts. | `lookup_quote_history`, `calculate_quote` | `search_quote_history` |
| **Sales** | Checks each priced line against stock and the delivery deadline, records the sale, and returns the customer summary. | `verify_order_feasibility`, `finalize_order` | `get_stock_level`, `create_transaction` |

All seven required helper functions are used by at least one tool: `create_transaction`, `get_all_inventory`, `get_stock_level`, `get_supplier_delivery_date`, `get_cash_balance`, `generate_financial_report` and `search_quote_history`. `get_all_inventory`, `get_cash_balance` and `generate_financial_report` are assigned to the Inventory agent, in `inventory_health_check`, as an internal health check.

### How the architecture was decided

- **Four agents, one per business function.** Inventory, quoting and sales are the three functions the brief asks for, and a fourth agent coordinates them. Keeping each worker to one job made the responsibilities easy to separate in the diagram and in the code.
- **Hard rules live in code, judgment lives in the model.** Stock checks, delivery deadlines, the cash reserve, the discount tiers and the size limits are plain Python inside the tools. The models do what they are good at: understanding the customer's wording, matching it to catalog items, and choosing the next tool. This keeps the system correct even when a small model (`gpt-4o-mini`) makes a poor choice, because a bad choice is refused by the tool.
- **Typed shared state.** A `RequestState` object carries the order lines, availability, quote and sale summary between agents. Agents never have to copy numbers to each other, which removes a whole class of transcription errors.
- **A fixed delegation order that is enforced.** The orchestrator cannot ask for a quote before stock has been checked, or finalize a sale before a quote exists. Those gates are in the code, because the model can issue several tool calls in one step and they would otherwise run at the same time.
- **The customer reply is built from recorded facts.** The model writes one opening and one closing sentence (rejected if it contains digits or prices). Items, prices, discounts, dates and reasons come from the recorded state. This guarantees the reply matches the database and cannot reveal internal information.

### Business rules and assumptions

| Rule | Value |
|---|---|
| Volume discount | 5% from 500 units, 10% from 1,000, 15% from 5,000 (total units quoted) |
| Extra history-based discount | Up to 3%, chosen by the quoting agent when similar past quotes justify it; always explained to the customer |
| Supplier cost | 70% of list price (assumption, never shown to customers) |
| Cash reserve | A purchase is refused if it would leave less than $5,000 |
| Single-line limit | Lines over 20,000 units are declined as too large for a standard order |
| Sizes | The catalog stocks A4 only among A sizes, so A3, A5 and similar sizes are declined |
| Reams | 1 ream = 500 sheets, converted in code |
| Delivery | A restock arrives on the date given by `get_supplier_delivery_date`; the sale is recorded on the date the goods are ready |

## 3. Evaluation results

The full set of 20 requests in `quote_requests_sample.csv` was run, and the results are in `test_results.csv` (columns: request id, request date, cash balance, inventory value, status, response).

**Summary**

| Measure | Result | Requirement |
|---|---|---|
| Requests processed | 20 of 20, no errors | full sample set |
| Fully fulfilled | 6 | at least 3 fulfilled |
| Partially fulfilled | 11 | not required |
| Not fulfilled | 3 | at least 1 declined, with a reason |
| Requests where the cash balance changed | 11 | at least 3 |
| Order lines confirmed / declined | 34 / 26 | |
| Customer replies containing internal information | 0 (checked for costs, cash, reserve, tool and error terms) | none allowed |

Cash moved from $45,059.70 to $45,693.58 and inventory value from $4,940.30 to $4,657.15 as of the last request date, so total assets rose from $50,000.00 to $50,350.73. A further $1,155.50 of confirmed sales are dated after that report (ship dates of 2025-04-19 and 2025-04-24, because those items had to be restocked first), so they are in the database but not in the final report.

**Per-request results**

| Request | Date | Status | Lines confirmed | Lines declined | Cash after request |
|---|---|---|---|---|---|
| 1 | 2025-04-01 | fulfilled | 3 | 0 | $45,122.75 |
| 2 | 2025-04-03 | partially fulfilled | 2 | 1 | $45,122.75 |
| 3 | 2025-04-04 | partially fulfilled | 1 | 2 | $45,122.75 |
| 4 | 2025-04-05 | fulfilled | 2 | 0 | $45,202.72 |
| 5 | 2025-04-05 | fulfilled | 3 | 0 | $45,246.22 |
| 6 | 2025-04-06 | fulfilled | 3 | 0 | $45,246.22 |
| 7 | 2025-04-07 | partially fulfilled | 3 | 1 | $45,496.02 |
| 8 | 2025-04-07 | partially fulfilled | 3 | 1 | $45,496.02 |
| 9 | 2025-04-07 | not fulfilled | 0 | 3 | $45,496.02 |
| 10 | 2025-04-08 | fulfilled | 2 | 0 | $45,488.92 |
| 11 | 2025-04-08 | partially fulfilled | 1 | 1 | $45,488.92 |
| 12 | 2025-04-08 | fulfilled | 3 | 0 | $45,488.92 |
| 13 | 2025-04-08 | not fulfilled | 0 | 2 | $45,488.92 |
| 14 | 2025-04-09 | partially fulfilled | 1 | 2 | $45,428.21 |
| 15 | 2025-04-12 | not fulfilled | 0 | 3 | $45,537.88 |
| 16 | 2025-04-13 | partially fulfilled | 2 | 1 | $45,623.29 |
| 17 | 2025-04-14 | partially fulfilled | 1 | 4 | $45,689.79 |
| 18 | 2025-04-14 | partially fulfilled | 1 | 2 | $45,689.79 |
| 19 | 2025-04-15 | partially fulfilled | 1 | 2 | $45,703.38 |
| 20 | 2025-04-17 | partially fulfilled | 2 | 1 | $45,693.58 |

**Why lines were declined** (26 declined lines in total)

| Reason given to the customer | Lines |
|---|---|
| Restock could not arrive before the requested date | 13 |
| Size not stocked (A3, A5) | 8 |
| Product not part of the current range | 5 |

### Strengths

- **Rejects what is impossible, with a reason.** Request 13 asked for delivery by April 10 on April 8. Both lines needed a restock arriving April 12, so the whole order was declined and the customer was told the earliest date. Request 15 was declined entirely: 10,000 sheets of A4 cannot arrive before April 19, A3 is not stocked, and cardboard is outside the range.
- **Accepts what is possible, including large restocked orders.** Request 8 combined three lines that all needed supplier restocks (matte, glossy and recycled paper, 4,500 units). They were bought, priced with a 10% volume discount plus an explained 3% history-based discount, and confirmed with exact ready dates.
- **Partial fulfilment instead of all-or-nothing.** In 11 requests the system supplied the lines it could and declined the rest, each with its own reason, which keeps sales that a strict all-or-nothing rule would lose.
- **Transparent customer replies.** Each reply lists confirmed items with prices, states why a discount was given, gives the ready date against the customer's requested date, and gives a specific reason for each declined line.
- **Safe with the money.** Every purchase is checked against the deadline and the cash reserve, restock quantities are capped at the recommended amount, finalizing a sale twice records it once, and the post-sale review counts stock already on order, so the same item is not bought repeatedly. No zero-unit orders appear in the database.

### Weaknesses seen in the results

- **Strict sizes lose sales.** Eight lines were declined because the customer wrote A3 or A5. The system could instead offer the nearest A4 product, but it never substitutes silently.
- **Tight deadlines dominate the rejections.** Thirteen lines failed only because supplier lead times are 1, 4 or 7 days by quantity. A split delivery (ship the part that fits the deadline) would recover some of these.
- **All-or-nothing per line.** A line asking for more than we can restock in time is declined entirely, even if part of it could ship.
- **Small wording issues.** Some declined lines repeat the quantity in the product text (for example "2,000 x 2000 sheets of A5 colored paper"), because the model copies the customer's phrase.
- **The 70% supplier cost, the $5,000 reserve and the discount tiers are assumptions,** not company data.
- **Item matching is by model judgment.** The code blocks unsafe matches (sizes, oversized quantities), but a plausible wrong match of a product name between two catalog items can still happen; for example, a customer's "poster boards" could be matched to either poster item.

## 4. Issues found and fixed during testing

Running the whole sample set several times exposed problems that a single request did not. They are recorded here because they shaped the final design.

1. **Invented customer replies.** An early version let the orchestrator model write the whole reply. It stated confirmed items and totals that were never sold, and gave vague reasons. Fix: replies are now built from recorded state.
2. **Repeated reorders.** The post-sale review bought the same item again and again, because deliveries on order were not counted. Fix: the review uses projected stock including inbound deliveries.
3. **Over-promising stock.** Stock committed to earlier customers' future-dated sales was ignored, so the system bought stock it could not then ship. Fix: availability uses the lower of current and projected stock.
4. **Wrong substitution.** "A3 paper" was matched to "A4 paper". Fix: a code check on sizes, plus a prompt rule.
5. **Concurrent tool calls.** The orchestrator issued the quote and sales steps in the same turn, so they ran at the same time. Fix: enforced ordering in the code.
6. **An absurd order.** "300 poster boards" was turned into 150,000 sheets (the model misapplied the ream conversion), and the system bought and sold them. Fix: reams are converted in code from a flag, and lines over 20,000 units are declined.
7. **Code review fixes.** A review of the file led to further fixes: restocks attaching to the wrong line when two lines share an item, a customer being told "failed" after a sale had been recorded, an unparseable deadline aborting a request, restocked items missing from the financial report, and a cash check that ignored later-dated purchases.

Known limitation left as is: the size filter reads the whole product description, so one line that mixes A4 and A3 text would be declined as a whole.

## 5. Suggested improvements

1. **Offer alternatives instead of only declining.** When a size or product is not stocked, or a deadline cannot be met, the quoting agent could propose the closest in-stock item (for example A4 for A3) or a split delivery, and let the customer choose. This turns many of the 26 declined lines into possible sales.
2. **A business advisor agent.** An additional (fifth) agent could analyze the transactions after each batch and recommend changes: which items to stock permanently because they are often requested and restocked (cardstock was restocked 11 times), better reorder thresholds, and discount tiers tuned to what customers accepted.
3. **Smarter quote history.** Past quotes are found by keyword matching, so more than two keywords return nothing. Matching on order size, event type and item mix (or using embeddings) would give the quoting agent better comparisons.
4. **Plan restocks by lead time.** Supplier lead time grows with quantity, which is why large restocks missed deadlines. Ordering slightly ahead for popular items, or splitting a large restock into batches, would shorten lead times.
5. **A customer agent for negotiation.** A simulated customer agent could use the sample context (job, event) to push back on prices or dates and test how the team responds.

## 6. How to run

1. Python 3.11 or later (developed and tested on 3.13); install the packages in `requirements.txt` (it includes `pydantic-ai`).
2. Put `UDACITY_OPENAI_API_KEY` in a `.env` file next to the script (the key is never stored in the code).
3. Run `python project_starter.py`. It rebuilds the database, processes all 20 requests, prints the agent actions and each reply, and writes `test_results.csv`. A full run takes roughly 15 minutes.
