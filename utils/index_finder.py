class TextSearchTool:
    """
    A utility class for searching terms within a text.
    This tool finds the first occurrence of a given search term in the provided text
    and returns the start and end indices of the term.
    """

    def __init__(self, text):
        """
        Initializes the TextSearchTool with the text to search within.
        """
        self.text = text

    def find_first_occurrence(self, search_term):
        """
        Finds the first occurrence of the specified search term in the text.
        """
        start_index = self.text.find(search_term)
        if start_index != -1:  # Found the term
            end_index = start_index + len(search_term) - 1
            return start_index, end_index
        else:
            return -1, -1  # Indicates the term was not found

# Example usage
if __name__ == "__main__":
    text = "I want to jump when I pose thumb down"
    search_tool = TextSearchTool(text)
    search_term = "thumb down"
    start, end = search_tool.find_first_occurrence(search_term)
    print(f"First occurrence: Start Index = {start}, End Index = {end + 1}")
