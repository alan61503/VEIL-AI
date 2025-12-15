def sync_to_cloud(vehicle_data):
    # For now, we’ll just simulate cloud sync
    if vehicle_data["exit_time"] is not None:
        print(f"Syncing to cloud: {vehicle_data['plate']}")
        # TODO: Add Firebase REST API or Supabase logic here
        return True
    return False
