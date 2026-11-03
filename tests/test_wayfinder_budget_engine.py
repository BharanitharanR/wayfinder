import unittest

from mesh.orchestrator import budget_engine


class HouseholdCashFlowTests(unittest.TestCase):
    def test_monthly_profile_reports_income_expenses_and_conservative_surplus(self):
        rows = [
            {"date": "2026-01-05", "amount": 100000, "direction": "credit", "category": "SALARY", "subcategory": "salary"},
            {"date": "2026-01-08", "amount": 30000, "direction": "debit", "category": "ESSENTIAL", "subcategory": "housing"},
            {"date": "2026-01-12", "amount": 10000, "direction": "debit", "category": "DISCRETIONARY", "subcategory": "dining"},
            {"date": "2026-01-15", "amount": 5000, "direction": "debit", "category": "OBLIGATION", "subcategory": "debt_repayment"},
            {"date": "2026-01-20", "amount": 10000, "direction": "debit", "category": "TRANSFER", "subcategory": "own_account_transfer"},
            {"date": "2026-01-21", "amount": 5000, "direction": "debit", "category": "INVESTMENT", "subcategory": "investment_contribution"},
            {"date": "2026-02-05", "amount": 100000, "direction": "credit", "category": "SALARY", "subcategory": "salary"},
            {"date": "2026-02-08", "amount": 35000, "direction": "debit", "category": "ESSENTIAL", "subcategory": "housing"},
            {"date": "2026-02-12", "amount": 15000, "direction": "debit", "category": "DISCRETIONARY", "subcategory": "dining"},
            {"date": "2026-02-15", "amount": 5000, "direction": "debit", "category": "OBLIGATION", "subcategory": "debt_repayment"},
        ]

        profile = budget_engine.build_profile(rows)

        self.assertEqual(profile["average_monthly_income"], 100000.0)
        self.assertEqual(profile["average_monthly_essential_expenses"], 32500.0)
        self.assertEqual(profile["average_monthly_discretionary_expenses"], 12500.0)
        self.assertEqual(profile["average_monthly_obligations"], 5000.0)
        self.assertEqual(profile["average_monthly_surplus"], 50000.0)
        self.assertEqual(profile["median_monthly_surplus"], 50000.0)
        self.assertEqual(profile["conservative_monthly_surplus_p25"], 45000.0)
        self.assertEqual(profile["classified_debit_ratio"], 1.0)
        self.assertTrue(profile["trustworthy"])
        self.assertNotIn("transfer", profile["classified_expense_totals"])
        self.assertNotIn("investment_contribution", profile["classified_expense_totals"])

    def test_safe_travel_budget_requires_every_input(self):
        result = budget_engine.calculate_safe_travel_budget(
            current_liquid_balance=300000,
            essential_monthly_expenses=50000,
            reserve_months=3,
            upcoming_obligations=20000,
            conservative_monthly_surplus=45000,
            months_until_departure=4,
            savings_capture_rate=0.4,
        )
        self.assertEqual(result["status"], "CALCULATED")
        self.assertEqual(result["emergency_reserve"], 150000.0)
        self.assertEqual(result["available_liquid_savings"], 130000.0)
        self.assertEqual(result["future_travel_contribution"], 72000.0)
        self.assertEqual(result["safe_travel_budget"], 202000.0)

        incomplete = budget_engine.calculate_safe_travel_budget(
            current_liquid_balance=None,
            essential_monthly_expenses=50000,
            reserve_months=3,
            upcoming_obligations=20000,
            conservative_monthly_surplus=45000,
            months_until_departure=4,
            savings_capture_rate=0.4,
        )
        self.assertEqual(incomplete["status"], "INSUFFICIENT_DATA")
        self.assertIn("current_liquid_balance", incomplete["missing_inputs"])

    def test_affordability_gate_is_deterministic(self):
        result = budget_engine.evaluate_trip_affordability(
            estimated_trip_cost=210000,
            safe_trip_budget=202000,
            search_exhausted=True,
        )
        self.assertEqual(result["status"], "NOT_AFFORDABLE")
        self.assertFalse(result["can_recommend_as_affordable"])


if __name__ == "__main__":
    unittest.main()
