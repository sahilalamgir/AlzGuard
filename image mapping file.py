for i in range(381, 706):
    if i == 1:
        print("mild")
    else:
        print(f"mild_{i}")
    
    # Pause every 50 lines
    if i % 200 == 0:
        input("Press Enter to continue...")