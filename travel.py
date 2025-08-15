import re
import json
from typing import Dict, List, Optional, Tuple
from autogen import ConversableAgent, LLMConfig


class TravelDataProvider:
    """Provides mock travel data simulating real APIs."""
    
    FLIGHT_DATABASE = {
        ("New York", "Tokyo"): [
            {"airline": "United Airlines", "price": 1850, "duration": "14h 20m", "stops": "Direct"},
            {"airline": "ANA", "price": 2100, "duration": "13h 45m", "stops": "Direct"},
            {"airline": "Korean Air", "price": 1320, "duration": "19h 10m", "stops": "1 stop (Seoul)"},
        ],
        ("London", "Paris"): [
            {"airline": "British Airways", "price": 180, "duration": "1h 25m", "stops": "Direct"},
            {"airline": "Air France", "price": 165, "duration": "1h 20m", "stops": "Direct"},
            {"airline": "Lufthansa", "price": 220, "duration": "3h 45m", "stops": "1 stop (Frankfurt)"},
        ]
    }
    
    HOTEL_DATABASE = {
        "Tokyo": [
            {"name": "Park Hyatt Tokyo", "price_per_night": 450, "rating": 4.8, "location": "Shinjuku"},
            {"name": "Tokyo Station Hotel", "price_per_night": 95, "rating": 4.2, "location": "Tokyo Station"},
            {"name": "Capsule Inn Akihabara", "price_per_night": 35, "rating": 3.8, "location": "Akihabara"},
        ],
        "Paris": [
            {"name": "Le Meurice", "price_per_night": 950, "rating": 4.9, "location": "1st Arrondissement"},
            {"name": "Hotel des Grands Boulevards", "price_per_night": 180, "rating": 4.3, "location": "2nd Arr."},
            {"name": "Generator Paris", "price_per_night": 45, "rating": 4.1, "location": "10th Arr."},
        ]
    }
    
    ACTIVITY_DATABASE = {
        "Tokyo": [
            {"name": "Senso-ji Temple", "price": 0, "category": "Cultural"},
            {"name": "Tokyo Skytree", "price": 25, "category": "Sightseeing"},
            {"name": "Tsukiji Market Tour", "price": 45, "category": "Food"},
            {"name": "teamLab Borderless", "price": 35, "category": "Art"},
        ],
        "Paris": [
            {"name": "Louvre Museum", "price": 17, "category": "Museum"},
            {"name": "Eiffel Tower", "price": 29, "category": "Sightseeing"},
            {"name": "Seine River Cruise", "price": 15, "category": "Sightseeing"},
            {"name": "Versailles Palace", "price": 20, "category": "Historical"},
        ]
    }
    
    @classmethod
    def get_flights(cls, origin: str, destination: str) -> List[Dict]:
        """Get available flights for a route"""
        route = (origin, destination)
        return cls.FLIGHT_DATABASE.get(route, [
            {"airline": "Generic Airways", "price": 800, "duration": "8h", "stops": "Direct"}
        ])
    
    @classmethod
    def get_hotels(cls, destination: str) -> List[Dict]:
        """Get available hotels for a destination"""
        return cls.HOTEL_DATABASE.get(destination, [
            {"name": f"Hotel {destination}", "price_per_night": 120, "rating": 4.0, "location": "City Center"}
        ])
    
    @classmethod
    def get_activities(cls, destination: str) -> List[Dict]:
        """Get available activities for a destination"""
        return cls.ACTIVITY_DATABASE.get(destination, [
            {"name": f"{destination} City Tour", "price": 30, "category": "Sightseeing"}
        ])


class TravelPlannerAG2:
    """
    Travel planner class that coordinates multiple agents.
    """
    
    def __init__(self, budget: float, origin: str, destination: str, nights: int):
        """Initialize the travel planner with trip parameters"""
        self.budget = budget
        self.origin = origin.strip()
        self.destination = destination.strip()
        self.nights = nights
        self.data_provider = TravelDataProvider()
        
        if budget <= 0:
            raise ValueError("Budget must be positive")
        if nights <= 0:
            raise ValueError("Number of nights must be positive")
        if not self.origin or not self.destination:
            raise ValueError("Origin and destination are required")
        
        self.llm_config = LLMConfig(
            model="mistral", api_key="", base_url="http://localhost:11434/v1",
            temperature=0.3, price=[0, 0]
        )
    
    def create_flight_agent(self) -> ConversableAgent:
        """Creates a specialized flight booking agent."""

        system_message = f"""
        You are a flight booking specialist for trips from {self.origin} to {self.destination}.
        Budget: ${self.budget:,.0f}
        
        When given flight options, analyze them and respond in this EXACT format:
        
        FLIGHT RECOMMENDATIONS:
        Premium: [Airline] - $[price] ([duration], [stops])
        Standard: [Airline] - $[price] ([duration], [stops]) 
        Budget: [Airline] - $[price] ([duration], [stops])
        
        BEST CHOICE: [Premium/Standard/Budget]
        REASON: [Brief explanation]
        
        Always recommend the option that best balances price, duration, and convenience within budget.
        """
        
        return ConversableAgent(
            name="FlightAgent",
            system_message=system_message,
            llm_config=self.llm_config,
            human_input_mode="NEVER", 
            max_consecutive_auto_reply=1
        )
    
    def create_hotel_agent(self) -> ConversableAgent:
        """Creates a specialized hotel booking agent."""

        system_message = f"""
        You are a hotel booking specialist for {self.destination}.
        Trip length: {self.nights} nights
        Total budget: ${self.budget:,.0f}
        
        When given hotel options, analyze them and respond in this EXACT format:
        
        HOTEL RECOMMENDATIONS:
        Luxury: [Hotel Name] - $[price per night] ([location], {self.nights} nights = $[total])
        Mid-Range: [Hotel Name] - $[price per night] ([location], {self.nights} nights = $[total])
        Budget: [Hotel Name] - $[price per night] ([location], {self.nights} nights = $[total])
        
        BEST CHOICE: [Luxury/Mid-Range/Budget]
        REASON: [Brief explanation considering location and value]
        
        Consider total cost for {self.nights} nights and recommend based on best value.
        """
        
        return ConversableAgent(
            name="HotelAgent",
            system_message=system_message,
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1
        )
    
    def create_activity_agent(self) -> ConversableAgent:
        """Creates a specialized activities planning agent."""

        system_message = f"""
        You are an activities specialist for {self.destination}.
        Trip length: {self.nights} days
        Budget: ${self.budget:,.0f}
        
        When given activity options, create a {self.nights}-day plan in this EXACT format:
        
        ACTIVITY PLAN:
        Day 1: [Activity 1] + [Activity 2] - Cost: $[total]
        Day 2: [Activity 1] + [Activity 2] - Cost: $[total]
        Day 3: [Activity 1] + [Activity 2] - Cost: $[total]
        
        TOTAL ACTIVITIES: $[sum of all days]
        HIGHLIGHTS: [Top 3 must-see activities]
        
        Mix free and paid activities. Prioritize must-see attractions while staying within budget.
        """
        
        return ConversableAgent(
            name="ActivityAgent", 
            system_message=system_message,
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1
        )
    
    def get_agent_recommendation(self, agent: ConversableAgent, data: List[Dict], context: str) -> str:
        """ Send data to an agent and get their recommendation. """

        try:
            # Format the data for the agent
            formatted_data = self._format_data_for_agent(data)
            message = f"{context}\n\n{formatted_data}"
            
            # Get agent's response
            response = agent.generate_reply(
                messages=[{"content": message, "role": "user"}]
            )
            
            # Handle different response types from the llm agent
            if isinstance(response, dict):
                return response.get("content", "No response generated")
            elif isinstance(response, str):
                return response
            else:
                return str(response)
                
        except Exception as e:
            print(f"Error getting recommendation from {agent.name}: {e}")
            return f"Error: Unable to get recommendation from {agent.name}"
    
    def _format_data_for_agent(self, data: List[Dict]) -> str:
        """Format data in a readable way for agents"""

        if not data:
            return "No data available"
        
        formatted = "Available Options:\n"
        for i, item in enumerate(data, 1):
            formatted += f"{i}. {item}\n"
        return formatted
    
    def _extract_costs_from_response(self, response: str, response_type: str) -> Tuple[int, str]:
        """Extract recommended option and its cost from agent response."""

        cost = 0
        details = "No details available"
        
        try:
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            
            # Find the recommended option
            recommendation = ""
            for line in lines:
                if line.startswith("BEST CHOICE:"):
                    recommendation = line.replace("BEST CHOICE:", "").strip().lower()
                    break
            
            # Extract cost based on recommendation and response type
            if response_type == "flight":
                for line in lines:
                    if recommendation in line.lower() and "$" in line:
                        price_match = re.search(r'\$(\d+)', line)
                        if price_match:
                            cost = int(price_match.group(1))
                            details = line
                            break
            
            elif response_type == "hotel":
                for line in lines:
                    if recommendation in line.lower() and "$" in line:
                        # Try to find total cost first
                        total_match = re.search(r'nights = \$(\d+)', line)
                        if total_match:
                            cost = int(total_match.group(1))
                        else:
                            # Fallback to per-night price and calculate total
                            price_match = re.search(r'\$(\d+)', line)
                            if price_match:
                                cost = int(price_match.group(1)) * self.nights
                        details = line
                        break
            
            elif response_type == "activity":
                for line in lines:
                    if line.startswith("TOTAL ACTIVITIES:"):
                        price_match = re.search(r'\$(\d+)', line)
                        if price_match:
                            cost = int(price_match.group(1))
                            details = "Activity plan created"
                            break
        
        except Exception as e:
            print(f"Error extracting costs: {e}")
        
        return cost, details
    
    def plan_trip(self) -> Optional[Dict]:
        """ Main method that coordinates all agents to create a complete travel plan. """
        print(f"\nAI Travel Planner (AG2)")
        print(f"{self.origin} → {self.destination}")
        print(f"{self.nights} nights - ${self.budget:,.0f} budget")
        print("─" * 50)
        
        try:
            # Step 1: Create specialized agents
            print("Creating travel specialists...")
            flight_agent = self.create_flight_agent()
            hotel_agent = self.create_hotel_agent()
            activity_agent = self.create_activity_agent()
            
            # Step 2: Gather travel data
            print("Gathering travel options...")
            flights = self.data_provider.get_flights(self.origin, self.destination)
            hotels = self.data_provider.get_hotels(self.destination)
            activities = self.data_provider.get_activities(self.destination)
            
            print(f"Found {len(flights)} flights, {len(hotels)} hotels, {len(activities)} activities")
            
            # Step 3: Get recommendations from each agent
            print("Consulting specialists...")
            
            flight_response = self.get_agent_recommendation(
                flight_agent,
                flights,
                f"Please recommend flights from {self.origin} to {self.destination}"
            )
            
            hotel_response = self.get_agent_recommendation(
                hotel_agent,
                hotels,
                f"Please recommend hotels in {self.destination} for {self.nights} nights"
            )
            
            activity_response = self.get_agent_recommendation(
                activity_agent,
                activities,
                f"Please create a {self.nights}-day activity plan for {self.destination}"
            )
            
            # Step 4: Extract costs and details
            flight_cost, flight_details = self._extract_costs_from_response(flight_response, "flight")
            hotel_cost, hotel_details = self._extract_costs_from_response(hotel_response, "hotel") 
            activity_cost, activity_details = self._extract_costs_from_response(activity_response, "activity")
            
            # Step 5: Calculate totals
            total_cost = flight_cost + hotel_cost + activity_cost
            remaining = self.budget - total_cost
            
            # Step 6: Display results
            self._display_trip_summary(
                flight_response, hotel_response, activity_response,
                flight_cost, hotel_cost, activity_cost, total_cost, remaining
            )
            
            # Step 7: Return structured result
            return {
                "total_cost": total_cost,
                "remaining_budget": remaining,
                "within_budget": total_cost <= self.budget,
                "flight_cost": flight_cost,
                "hotel_cost": hotel_cost,
                "activity_cost": activity_cost,
                "flight_recommendation": flight_details,
                "hotel_recommendation": hotel_details,
                "activity_recommendation": activity_details
            }
            
        except Exception as e:
            print(f"Trip planning failed: {e}")
            return None
    
    def _display_trip_summary(self, flight_resp: str, hotel_resp: str, activity_resp: str,
                            flight_cost: int, hotel_cost: int, activity_cost: int,
                            total_cost: int, remaining: int):
        """Display a formatted trip summary"""
        
        print("\n" + "=" * 50)
        print("TRIP SUMMARY")
        print("=" * 50)
        
        print("FLIGHTS:")
        self._print_agent_section(flight_resp)
        
        print("\nHOTELS:")
        self._print_agent_section(hotel_resp)
        
        print("\nACTIVITIES:")
        self._print_agent_section(activity_resp)
        
        print(f"\nCOST BREAKDOWN:")
        print(f"   Flights:    ${flight_cost:,}")
        print(f"   Hotels:     ${hotel_cost:,}")  
        print(f"   Activities: ${activity_cost:,}")
        print(f"   ──────────────────────")
        print(f"   Total:      ${total_cost:,}")
        print(f"   Remaining:  ${remaining:,}")
        
        if total_cost <= self.budget:
            print(f"\nTRIP FITS BUDGET!")
            print(f" Ready for your {self.nights}-night adventure!")
        else:
            over_budget = total_cost - self.budget
            print(f"\n Over budget by ${over_budget:,}")
            print(f" Consider reducing accommodation or activities")
        
        print("=" * 50)
    
    def _print_agent_section(self, response: str):
        """Print a cleaned version of agent response"""
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        for line in lines:
            if any(keyword in line for keyword in ['Premium:', 'Standard:', 'Budget:', 'Luxury:', 'Mid-Range:', 'Day ', 'BEST CHOICE:', 'HIGHLIGHTS:']):
                if 'BEST CHOICE:' in line:
                    print(f"     {line}")
                else:
                    print(f"   {line}")


def main():
    """ Example usage of the travel planner. """
    
    try:
        # Create travel planner instance
        planner = TravelPlannerAG2(
            budget=3500,
            origin="New York", 
            destination="Tokyo",
            nights=3
        )
        
        # Plan the trip using agents
        result = planner.plan_trip()
        
        # Check results
        if result:
            if result['within_budget']:
                print(f"\n SUCCESS! Trip planned within budget")
                print(f" Total cost: ${result['total_cost']:,}")
                print(f" Money left: ${result['remaining_budget']:,}")
            else:
                print(f"\n Trip planned but over budget") 
                print(f" Consider adjusting your preferences")
        else:
            print(f"\n Unable to plan trip")
            print(f" Please check your setup and try again")
            
    except Exception as e:
        print(f" Error: {e}")

if __name__ == "__main__":
    main()