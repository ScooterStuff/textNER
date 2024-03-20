def find_first_occurrence(text, search_term):
    start_index = text.find(search_term)
    if start_index != -1:  # Found the term
        end_index = start_index + len(search_term) - 1
        return start_index, end_index
    else:
        return -1, -1  # Indicates the term was not found

# Example usage
text = "I want to jump when I pose thumb down"
search_term = "thumb down"
start, end = find_first_occurrence(text, search_term)
print(f"First occurrence: Start Index = {start}, End Index = {end+1}")


# Not Indepth Mode (This would assumes most of thing map to most of things)

# Mario - GAME
# left - ORITEINTATION
# hand - LANDMARK
# fist - POSES
# double pinch - GESTURE
# jump - ACTION