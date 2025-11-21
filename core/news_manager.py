"""
News Manager for Prop Firm Compliance
Blocks trading around high-impact news events.

CRITICAL: The5ers and other prop firms ban trading 2 minutes before/after
high-impact news. This module ensures compliance.
"""

import os
import csv
import logging
from datetime import datetime, timedelta
import pytz


class NewsManager:
    """
    Manages high-impact news events and trading blackouts.

    Usage:
        news_manager = NewsManager('inputs/blocked_times.csv')
        if news_manager.is_news_event():
            print("NEWS BLACKOUT - Cannot trade")
    """

    def __init__(self, blocked_times_file='inputs/blocked_times.csv', buffer_minutes=2):
        """
        Initialize News Manager

        Args:
            blocked_times_file: Path to CSV with blocked times
            buffer_minutes: Minutes before/after news to block (default: 2)
        """
        self.logger = logging.getLogger(__name__)
        self.buffer_minutes = buffer_minutes
        self.blocked_times = []

        # Load blocked times from file
        if os.path.exists(blocked_times_file):
            self._load_blocked_times(blocked_times_file)
        else:
            self.logger.warning(f"Blocked times file not found: {blocked_times_file}")
            self.logger.info("Creating default blocked times file...")
            self._create_default_file(blocked_times_file)

    def _load_blocked_times(self, filepath):
        """
        Load blocked times from CSV file

        CSV format:
        date,time,event,impact
        2025-11-20,08:30,NFP,HIGH
        2025-11-20,14:00,FOMC,HIGH
        """
        try:
            with open(filepath, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Parse date and time
                    date_str = row.get('date', '').strip()
                    time_str = row.get('time', '').strip()
                    event = row.get('event', 'Unknown').strip()
                    impact = row.get('impact', 'HIGH').strip()

                    # Only block HIGH impact events
                    if impact.upper() != 'HIGH':
                        continue

                    # Parse datetime
                    try:
                        if date_str:
                            dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
                        else:
                            # If no date, use today
                            today = datetime.now().strftime("%Y-%m-%d")
                            dt = datetime.strptime(f"{today} {time_str}", "%Y-%m-%d %H:%M")

                        self.blocked_times.append({
                            'datetime': dt,
                            'event': event,
                            'impact': impact
                        })
                    except ValueError as e:
                        self.logger.warning(f"Invalid time format in CSV: {date_str} {time_str} - {e}")

            self.logger.info(f"Loaded {len(self.blocked_times)} blocked news events")

        except Exception as e:
            self.logger.error(f"Error loading blocked times: {e}")

    def _create_default_file(self, filepath):
        """Create a default blocked times CSV file"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Default high-impact events (common times)
        default_events = [
            # Format: date, time (UTC), event, impact
            ('', '08:30', 'US Employment Data', 'HIGH'),
            ('', '10:00', 'US ISM Manufacturing', 'HIGH'),
            ('', '13:30', 'US CPI/PPI', 'HIGH'),
            ('', '14:00', 'FOMC Rate Decision', 'HIGH'),
            ('', '14:30', 'FOMC Press Conference', 'HIGH'),
            ('', '18:00', 'FOMC Minutes', 'HIGH'),
        ]

        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['date', 'time', 'event', 'impact'])
                for event in default_events:
                    writer.writerow(event)

            self.logger.info(f"Created default blocked times file: {filepath}")
            self.logger.info("Please update with actual news times for your trading day")

        except Exception as e:
            self.logger.error(f"Error creating default file: {e}")

    def is_news_event(self, current_time=None):
        """
        Check if current time is within buffer of any blocked news event

        Args:
            current_time: Time to check (default: now)

        Returns:
            bool: True if in news blackout period
        """
        if current_time is None:
            current_time = datetime.now()

        # Check against each blocked time
        for event in self.blocked_times:
            event_time = event['datetime']

            # Only check events for today (or matching date)
            if event_time.date() != current_time.date():
                # If no date was specified in CSV, check time only
                if event['datetime'].year == datetime.now().year:
                    event_time = event_time.replace(
                        year=current_time.year,
                        month=current_time.month,
                        day=current_time.day
                    )
                else:
                    continue

            # Calculate time difference
            time_diff = abs((current_time - event_time).total_seconds() / 60)

            # Check if within buffer
            if time_diff <= self.buffer_minutes:
                self.logger.warning(
                    f"NEWS BLACKOUT: {event['event']} at {event_time.strftime('%H:%M')} "
                    f"({time_diff:.1f} min away)"
                )
                return True

        return False

    def get_next_event(self, current_time=None):
        """
        Get the next upcoming news event

        Args:
            current_time: Time to check from (default: now)

        Returns:
            dict: Next event info or None
        """
        if current_time is None:
            current_time = datetime.now()

        upcoming = []
        for event in self.blocked_times:
            event_time = event['datetime']

            # Adjust for today if no date specified
            if event_time.year == datetime.now().year:
                event_time = event_time.replace(
                    year=current_time.year,
                    month=current_time.month,
                    day=current_time.day
                )

            if event_time > current_time:
                upcoming.append({
                    'datetime': event_time,
                    'event': event['event'],
                    'minutes_until': (event_time - current_time).total_seconds() / 60
                })

        if upcoming:
            # Return the soonest event
            return min(upcoming, key=lambda x: x['minutes_until'])

        return None

    def add_blocked_time(self, time_str, event_name='Manual Block', date_str=None):
        """
        Manually add a blocked time

        Args:
            time_str: Time in HH:MM format
            event_name: Name of the event
            date_str: Date in YYYY-MM-DD format (default: today)
        """
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        try:
            dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
            self.blocked_times.append({
                'datetime': dt,
                'event': event_name,
                'impact': 'HIGH'
            })
            self.logger.info(f"Added blocked time: {event_name} at {dt}")
        except ValueError as e:
            self.logger.error(f"Invalid time format: {e}")

    def clear_old_events(self):
        """Remove events from previous days"""
        today = datetime.now().date()
        self.blocked_times = [
            event for event in self.blocked_times
            if event['datetime'].date() >= today
        ]

    def get_status(self):
        """
        Get current news status

        Returns:
            dict: Status information
        """
        current_time = datetime.now()
        is_blocked = self.is_news_event(current_time)
        next_event = self.get_next_event(current_time)

        return {
            'is_blocked': is_blocked,
            'current_time': current_time.strftime('%H:%M'),
            'buffer_minutes': self.buffer_minutes,
            'total_events': len(self.blocked_times),
            'next_event': next_event
        }


# Convenience function
def check_news_blackout(blocked_times_file='inputs/blocked_times.csv', buffer_minutes=2):
    """
    Quick check if currently in news blackout

    Args:
        blocked_times_file: Path to blocked times CSV
        buffer_minutes: Buffer around news events

    Returns:
        bool: True if in blackout period
    """
    manager = NewsManager(blocked_times_file, buffer_minutes)
    return manager.is_news_event()


if __name__ == "__main__":
    # Test the news manager
    logging.basicConfig(level=logging.INFO)

    manager = NewsManager()

    # Add some test events
    manager.add_blocked_time("14:30", "Test FOMC")
    manager.add_blocked_time("08:30", "Test NFP")

    # Check status
    status = manager.get_status()
    print(f"\nNews Manager Status:")
    print(f"  Is Blocked: {status['is_blocked']}")
    print(f"  Current Time: {status['current_time']}")
    print(f"  Total Events: {status['total_events']}")

    if status['next_event']:
        print(f"  Next Event: {status['next_event']['event']} "
              f"in {status['next_event']['minutes_until']:.1f} min")
