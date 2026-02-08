"""
Capacity Planning System for OpenClaw AI Agent System
Forecasts resource needs and plans for growth
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import statistics
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CapacityPlan:
    """Capacity plan for a specific time period"""
    period_start: datetime
    period_end: datetime
    predicted_users: int
    predicted_concurrent_tasks: int
    required_instances: int
    required_cpu_cores: float
    required_memory_gb: float
    required_storage_gb: float
    required_network_mbps: float
    required_gpt_tokens_per_min: int
    confidence: float
    assumptions: List[str] = field(default_factory=list)


@dataclass
class GrowthProjection:
    """Growth projection over time"""
    timestamp: datetime
    metric_name: str
    current_value: float
    projected_value: float
    growth_rate: float
    confidence: float


class CapacityPlanner:
    """
    Capacity planning system for forecasting resource needs
    
    Features:
    - Historical trend analysis
    - Growth projections
    - Resource requirement calculations
    - Cost estimation
    """
    
    # Base resource requirements per agent instance
    BASE_REQUIREMENTS = {
        "cpu_cores": 2.0,
        "memory_gb": 4.0,
        "disk_gb": 10.0,
        "network_mbps": 100.0,
    }
    
    # GPT-5.2 specific overhead (high thinking mode)
    GPT52_OVERHEAD = {
        "cpu_multiplier": 1.5,
        "memory_multiplier": 2.0,
        "token_rate_per_min": 1000,
    }
    
    # User capacity per instance (conservative for GPT-5.2)
    USERS_PER_INSTANCE = 5
    
    # Peak factor for handling spikes
    DEFAULT_PEAK_FACTOR = 1.5
    
    def __init__(self, metrics_collector=None):
        self.metrics_collector = metrics_collector
        
        # Historical data
        self.user_growth_history: deque = deque(maxlen=365)  # 1 year daily data
        self.resource_usage_history: deque = deque(maxlen=365)
        
        # Growth rates (per period)
        self.growth_rates: Dict[str, float] = {
            "users_daily": 0.02,  # 2% daily growth
            "users_weekly": 0.15,  # 15% weekly growth
            "users_monthly": 0.50,  # 50% monthly growth
        }
        
        # Lock
        self._lock = threading.RLock()
        
        logger.info("CapacityPlanner initialized")
    
    def record_user_count(self, user_count: int, timestamp: datetime = None):
        """Record current user count for growth tracking"""
        if timestamp is None:
            timestamp = datetime.now()
        
        with self._lock:
            self.user_growth_history.append({
                "timestamp": timestamp,
                "user_count": user_count
            })
    
    def calculate_growth_rate(self, 
                             metric_history: deque,
                             days: int = 30) -> float:
        """
        Calculate growth rate from historical data
        
        Returns:
            Daily growth rate as decimal (e.g., 0.02 = 2%)
        """
        if len(metric_history) < 2:
            return self.growth_rates["users_daily"]
        
        # Get data for specified period
        cutoff = datetime.now() - timedelta(days=days)
        recent_data = [
            d for d in metric_history
            if d["timestamp"] >= cutoff
        ]
        
        if len(recent_data) < 2:
            return self.growth_rates["users_daily"]
        
        # Sort by timestamp
        recent_data.sort(key=lambda x: x["timestamp"])
        
        # Calculate growth rate using linear regression
        start_value = recent_data[0]["user_count"]
        end_value = recent_data[-1]["user_count"]
        days_elapsed = (recent_data[-1]["timestamp"] - 
                       recent_data[0]["timestamp"]).days
        
        if days_elapsed == 0 or start_value == 0:
            return 0.0
        
        # Compound daily growth rate
        total_growth = (end_value - start_value) / start_value
        daily_rate = (1 + total_growth) ** (1 / days_elapsed) - 1
        
        return daily_rate
    
    def project_users(self, 
                     current_users: int,
                     days_ahead: int,
                     use_historical: bool = True) -> Tuple[int, float]:
        """
        Project user count for N days ahead
        
        Returns:
            (projected_users, confidence)
        """
        if use_historical and self.user_growth_history:
            growth_rate = self.calculate_growth_rate(
                self.user_growth_history, days=30
            )
        else:
            growth_rate = self.growth_rates["users_daily"]
        
        # Project using compound growth
        projected = current_users * ((1 + growth_rate) ** days_ahead)
        
        # Confidence decreases with projection distance
        confidence = max(0.3, 1.0 - (days_ahead / 365))
        
        return int(projected), confidence
    
    def calculate_capacity_requirements(self,
                                       concurrent_users: int,
                                       peak_factor: float = None) -> Dict:
        """
        Calculate required capacity for given user count
        
        Args:
            concurrent_users: Expected concurrent users
            peak_factor: Multiplier for peak load (default: 1.5)
            
        Returns:
            Dict with capacity requirements
        """
        if peak_factor is None:
            peak_factor = self.DEFAULT_PEAK_FACTOR
        
        # Base instances needed
        base_instances = (concurrent_users + self.USERS_PER_INSTANCE - 1) // self.USERS_PER_INSTANCE
        
        # Apply peak factor
        peak_instances = int(base_instances * peak_factor)
        
        # Calculate resources with GPT-5.2 overhead
        total_cpu = (
            peak_instances * 
            self.BASE_REQUIREMENTS["cpu_cores"] * 
            self.GPT52_OVERHEAD["cpu_multiplier"]
        )
        
        total_memory = (
            peak_instances * 
            self.BASE_REQUIREMENTS["memory_gb"] * 
            self.GPT52_OVERHEAD["memory_multiplier"]
        )
        
        total_storage = peak_instances * self.BASE_REQUIREMENTS["disk_gb"]
        total_network = peak_instances * self.BASE_REQUIREMENTS["network_mbps"]
        
        # GPT-5.2 token requirements
        gpt_tokens = peak_instances * self.GPT52_OVERHEAD["token_rate_per_min"]
        
        return {
            "instances": peak_instances,
            "cpu_cores": round(total_cpu, 1),
            "memory_gb": round(total_memory, 1),
            "storage_gb": total_storage,
            "network_mbps": total_network,
            "gpt_tokens_per_min": gpt_tokens,
            "base_instances": base_instances,
            "peak_factor": peak_factor,
        }
    
    def create_capacity_plan(self,
                            current_users: int,
                            plan_period_days: int = 90,
                            review_interval_days: int = 30) -> List[CapacityPlan]:
        """
        Create capacity plan for specified period
        
        Args:
            current_users: Current user count
            plan_period_days: Total planning period
            review_interval_days: Interval between plan reviews
            
        Returns:
            List of CapacityPlan objects
        """
        plans = []
        now = datetime.now()
        
        # Create plans for each review interval
        num_intervals = plan_period_days // review_interval_days
        
        for i in range(num_intervals):
            interval_start = now + timedelta(days=i * review_interval_days)
            interval_end = now + timedelta(days=(i + 1) * review_interval_days)
            
            # Project users for end of interval
            days_ahead = (i + 1) * review_interval_days
            projected_users, confidence = self.project_users(
                current_users, days_ahead
            )
            
            # Estimate concurrent users (typically 20-30% of total)
            concurrent_users = int(projected_users * 0.25)
            
            # Calculate capacity requirements
            requirements = self.calculate_capacity_requirements(concurrent_users)
            
            # Create plan
            plan = CapacityPlan(
                period_start=interval_start,
                period_end=interval_end,
                predicted_users=projected_users,
                predicted_concurrent_tasks=concurrent_users * 2,  # Estimate
                required_instances=requirements["instances"],
                required_cpu_cores=requirements["cpu_cores"],
                required_memory_gb=requirements["memory_gb"],
                required_storage_gb=requirements["storage_gb"],
                required_network_mbps=requirements["network_mbps"],
                required_gpt_tokens_per_min=requirements["gpt_tokens_per_min"],
                confidence=confidence,
                assumptions=[
                    f"User growth rate: {self.growth_rates['users_daily']*100:.1f}% daily",
                    f"Peak factor: {self.DEFAULT_PEAK_FACTOR}x",
                    f"Users per instance: {self.USERS_PER_INSTANCE}",
                    f"Concurrent user ratio: 25%",
                ]
            )
            
            plans.append(plan)
        
        return plans
    
    def estimate_costs(self, 
                      capacity_requirements: Dict,
                      cost_per_cpu: float = 0.05,  # per hour
                      cost_per_gb_ram: float = 0.01,  # per hour
                      cost_per_gb_storage: float = 0.10,  # per month
                      cost_per_mbps: float = 0.01,  # per hour
                      cost_per_gpt_token: float = 0.00002) -> Dict:
        """
        Estimate monthly costs based on capacity requirements
        
        Args:
            capacity_requirements: Output from calculate_capacity_requirements
            cost_per_cpu: Cost per CPU core per hour
            cost_per_gb_ram: Cost per GB RAM per hour
            cost_per_gb_storage: Cost per GB storage per month
            cost_per_mbps: Cost per Mbps bandwidth per hour
            cost_per_gpt_token: Cost per GPT-5.2 token
            
        Returns:
            Dict with cost breakdown
        """
        hours_per_month = 730  # ~30 days
        
        # Compute costs
        compute_cost = (
            capacity_requirements["cpu_cores"] * cost_per_cpu +
            capacity_requirements["memory_gb"] * cost_per_gb_ram
        ) * hours_per_month
        
        # Storage cost
        storage_cost = (
            capacity_requirements["storage_gb"] * cost_per_gb_storage
        )
        
        # Network cost
        network_cost = (
            capacity_requirements["network_mbps"] * cost_per_mbps
        ) * hours_per_month
        
        # GPT-5.2 cost (assuming 50% utilization)
        gpt_tokens_monthly = (
            capacity_requirements["gpt_tokens_per_min"] * 
            60 * 24 * 30 * 0.5  # 50% utilization
        )
        gpt_cost = gpt_tokens_monthly * cost_per_gpt_token
        
        total_cost = compute_cost + storage_cost + network_cost + gpt_cost
        
        return {
            "compute_monthly": round(compute_cost, 2),
            "storage_monthly": round(storage_cost, 2),
            "network_monthly": round(network_cost, 2),
            "gpt52_monthly": round(gpt_cost, 2),
            "total_monthly": round(total_cost, 2),
            "total_annual": round(total_cost * 12, 2),
            "breakdown": {
                "compute_percent": round(compute_cost / total_cost * 100, 1),
                "storage_percent": round(storage_cost / total_cost * 100, 1),
                "network_percent": round(network_cost / total_cost * 100, 1),
                "gpt52_percent": round(gpt_cost / total_cost * 100, 1),
            }
        }
    
    def generate_growth_projections(self,
                                    current_users: int,
                                    metrics: List[str] = None) -> List[GrowthProjection]:
        """
        Generate growth projections for multiple metrics
        
        Args:
            current_users: Current user count
            metrics: List of metrics to project
            
        Returns:
            List of GrowthProjection objects
        """
        if metrics is None:
            metrics = ["users", "instances", "cpu_cores", "memory_gb"]
        
        projections = []
        now = datetime.now()
        
        # Project for 1, 3, 6, 12 months
        periods = [30, 90, 180, 365]
        
        for days in periods:
            projected_users, confidence = self.project_users(
                current_users, days
            )
            
            # Calculate derived metrics
            requirements = self.calculate_capacity_requirements(
                int(projected_users * 0.25)  # 25% concurrent
            )
            
            for metric in metrics:
                if metric == "users":
                    current = current_users
                    projected = projected_users
                elif metric in requirements:
                    current = self.calculate_capacity_requirements(
                        int(current_users * 0.25)
                    ).get(metric, 0)
                    projected = requirements[metric]
                else:
                    continue
                
                # Calculate growth rate
                if current > 0:
                    growth_rate = (projected - current) / current
                else:
                    growth_rate = 0
                
                projection = GrowthProjection(
                    timestamp=now + timedelta(days=days),
                    metric_name=metric,
                    current_value=current,
                    projected_value=projected,
                    growth_rate=growth_rate,
                    confidence=confidence
                )
                
                projections.append(projection)
        
        return projections
    
    def get_recommendations(self, 
                           current_capacity: Dict,
                           predicted_needs: Dict) -> List[Dict]:
        """
        Generate capacity recommendations
        
        Args:
            current_capacity: Current capacity configuration
            predicted_needs: Predicted capacity needs
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check each resource type
        resource_types = [
            "instances", "cpu_cores", "memory_gb", 
            "storage_gb", "network_mbps"
        ]
        
        for resource in resource_types:
            current = current_capacity.get(resource, 0)
            needed = predicted_needs.get(resource, 0)
            
            if needed > current * 1.2:  # 20% threshold
                recommendations.append({
                    "resource": resource,
                    "current": current,
                    "recommended": needed,
                    "urgency": "high" if needed > current * 1.5 else "medium",
                    "action": f"Increase {resource} from {current} to {needed}",
                    "timeline": "immediate" if needed > current * 1.5 else "30_days"
                })
            elif needed < current * 0.7:  # Over-provisioned
                recommendations.append({
                    "resource": resource,
                    "current": current,
                    "recommended": int(needed * 1.1),  # 10% buffer
                    "urgency": "low",
                    "action": f"Consider reducing {resource} to optimize costs",
                    "timeline": "next_review"
                })
        
        return recommendations
    
    def export_plan(self, 
                   plans: List[CapacityPlan],
                   format: str = "json") -> str:
        """Export capacity plan to various formats"""
        if format == "json":
            plan_data = []
            for plan in plans:
                plan_data.append({
                    "period_start": plan.period_start.isoformat(),
                    "period_end": plan.period_end.isoformat(),
                    "predicted_users": plan.predicted_users,
                    "predicted_concurrent_tasks": plan.predicted_concurrent_tasks,
                    "required_instances": plan.required_instances,
                    "required_cpu_cores": plan.required_cpu_cores,
                    "required_memory_gb": plan.required_memory_gb,
                    "required_storage_gb": plan.required_storage_gb,
                    "required_network_mbps": plan.required_network_mbps,
                    "required_gpt_tokens_per_min": plan.required_gpt_tokens_per_min,
                    "confidence": plan.confidence,
                    "assumptions": plan.assumptions,
                })
            return json.dumps(plan_data, indent=2)
        
        elif format == "markdown":
            lines = ["# Capacity Plan\n"]
            for i, plan in enumerate(plans):
                lines.append(f"## Period {i+1}: {plan.period_start.date()} to {plan.period_end.date()}\n")
                lines.append(f"- **Predicted Users**: {plan.predicted_users:,}")
                lines.append(f"- **Required Instances**: {plan.required_instances}")
                lines.append(f"- **CPU Cores**: {plan.required_cpu_cores}")
                lines.append(f"- **Memory**: {plan.required_memory_gb} GB")
                lines.append(f"- **Storage**: {plan.required_storage_gb} GB")
                lines.append(f"- **Network**: {plan.required_network_mbps} Mbps")
                lines.append(f"- **GPT Tokens/min**: {plan.required_gpt_tokens_per_min:,}")
                lines.append(f"- **Confidence**: {plan.confidence*100:.0f}%\n")
            return "\n".join(lines)
        
        return ""


# Example usage
if __name__ == "__main__":
    # Create planner
    planner = CapacityPlanner()
    
    # Simulate historical data
    current_users = 100
    for i in range(30):
        planner.record_user_count(
            int(current_users * (1.02 ** i)),
            datetime.now() - timedelta(days=30-i)
        )
    
    # Create capacity plan
    plans = planner.create_capacity_plan(
        current_users=current_users,
        plan_period_days=180,
        review_interval_days=30
    )
    
    print("=== CAPACITY PLAN ===\n")
    for i, plan in enumerate(plans):
        print(f"Period {i+1}: {plan.period_start.date()} to {plan.period_end.date()}")
        print(f"  Users: {plan.predicted_users:,}")
        print(f"  Instances: {plan.required_instances}")
        print(f"  CPU: {plan.required_cpu_cores} cores")
        print(f"  Memory: {plan.required_memory_gb} GB")
        print(f"  Storage: {plan.required_storage_gb} GB")
        print(f"  Network: {plan.required_network_mbps} Mbps")
        print(f"  Confidence: {plan.confidence*100:.0f}%\n")
    
    # Estimate costs
    costs = planner.estimate_costs(plans[0].__dict__)
    print("=== COST ESTIMATE ===")
    print(f"Monthly: ${costs['total_monthly']}")
    print(f"Annual: ${costs['total_annual']}")
    print(f"Breakdown: {costs['breakdown']}\n")
    
    # Generate projections
    projections = planner.generate_growth_projections(current_users)
    print("=== GROWTH PROJECTIONS ===")
    for p in projections[:4]:
        print(f"{p.metric_name}: {p.current_value} -> {p.projected_value} "
              f"({p.growth_rate*100:+.1f}%) by {p.timestamp.date()}")
    
    # Export plan
    print("\n=== JSON EXPORT ===")
    print(planner.export_plan(plans[:2], "json")[:500] + "...")
