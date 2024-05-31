from bs4 import BeautifulSoup
import sys
import requests
import json
import urllib.parse

# Replace with the actual URL of the drugs.com search page
root = "https://www.drugs.com/imprints.php?"  # Example URL


imprint = input("Enter imprint: ")
color_input = input("Enter color: ")
shape_input = input("Enter shape: ")
 



color_to_number = {
   
"beige": 14,
"black": 73,
"blue": 1,
"brown": 2,
"clear": 3,
"gold": 4,
"gray": 5,
"green": 6,
"maroon": 44,
"orange": 7,
"peach": 74,
"pink": 8,
"purple": 9,
"red": 10,
"tan": 11,
"white": 12,
"yellow": 13,			
"beige&red": 69,
"black&green": 55,
"black&teal": 70,
"black&yellow": 48,
"blue&brown": 52,
"blue&gray": 45,
"blue&green": 75,
"blue&orange": 71,
"blue&peach": 53,
"blue&pink": 34,
"blue&white": 19,
"blue&white specks": 26,
"blue&yellow": 21,
"brown&clear": 47,
"brown&orange": 54,
"brown&peach": 28,
"brown&red": 16,
"brown&white": 57,
"brown&yellow": 27,
"clear&green": 49,
"dark&light green": 46,
"gold&white": 51,
"gray&peach": 61,
"gray&pink": 39,
"gray&red": 58,
"gray&white": 67,
"gray&yellow": 68,
"green&orange": 65,
"green&peach": 63,
"green&pink": 56,
"green&purple": 43,
"green&turquoise": 62,
"green&white": 30,
"green&yellow": 22,
"lavender&white": 42,
"maroon&pink": 40,
"orange&turquoise": 50,
"orange&white": 64,
"orange&yellow": 23,
"peach&purple": 60,
"peach&red": 66,
"peach&white": 18,
"pink&purple": 15,
"pink&red specks": 37,
"pink&turquoise": 29,
"pink&white": 25,
"pink&yellow": 72,
"red&turquoise": 17,
"red&white": 35,
"red&yellow": 20,
"tan&white": 33,
"turquoise&white": 59,
"turquoise&yellow": 24,
"white&blue specks": 32,
"white&red specks": 41,
"white&yellow": 38,
"yellow&gray": 31,
"yellow&white": 36
	
	
}

shape_to_number = {
   
"barrel": 1,
"capsule": 5,
"oblong": 5,
"character-shape": 6,
"egg-shape": 9,
"eight-sided": 10,
"oval": 11,
"figure eight-shape": 12,
"five-sided": 13,
"four-sided": 14,
"gear-shape": 15,
"heart-shape": 16,
"kidney-shape": 18,
"rectangle": 23,
"round": 24,
"seven-sided": 25,
"six-sided": 27,
"three-sided": 32,
"u-shape": 33

}

def get_color_number(color):
  """Retrieves the number associated with a color from the dictionary.

  Args:
      color: The color name to look up (case-insensitive).

  Returns:
      The corresponding number from the dictionary, or None if not found.
  """

  # Convert input color to lowercase for case-insensitive matching
  color = color.lower()

  if color in color_to_number:
    col_to_num = color_to_number[color]
    return col_to_num
  else:
    return None

# Example usage
#color_input = input("Enter a color: ")
color_number = get_color_number(color_input)


# You can now pass the color_number to other functions
# ... (your code to use the color_number)
#Shape
def get_shape_number(shape):
  """Retrieves the number associated with a color from the dictionary.

  Args:
      color: The color name to look up (case-insensitive).

  Returns:
      The corresponding number from the dictionary, or None if not found.
  """

  # Convert input color to lowercase for case-insensitive matching
  shape = shape.lower()

  if shape in shape_to_number:
    return shape_to_number[shape]
  else:
    return None

# Example usage
#color_input = input("Enter a color: ")
shape_number = get_shape_number(shape_input)



encoded_imprint = urllib.parse.quote(imprint)
#encoded_color = urllib.parse.quote(col_to_num)
#encoded_shape = urllib.parse.quote(shape)


# Construct URL with proper encoding
url = f"{root}imprint={encoded_imprint}&color={color_number}&shape={shape_number}"
page = requests.get(url)
#soup =  BeautifulSoup(page.content, 'html.parser')
print(url)




if page.status_code == 200:  # Check for successful response
    soup = BeautifulSoup(page.content, 'html.parser')

    # Find relevant elements based on drugs.com's structure (replace with actual selectors)
    drug_results = soup.find('a', class_='ddc-btn ddc-btn-small', href=True)  # Get href if found
    if drug_results:
        link_url = drug_results['href']
        print(f"Link URL: {drug_results['href']}")
    else:
        print("No matching anchor tag found.")


    from bs4 import BeautifulSoup
    import requests


    # Construct the full URL (assuming a base URL)
    base_url = "https://www.drugs.com"  # Replace with the actual base URL if needed
    full_url = base_url + link_url
    print(full_url)

    # Open the linked webpage using requests
    response = requests.get(full_url)

    # Check for successful response (status code 200)
    if response.status_code == 200:
        # Parse the content of the linked webpage
        soup = BeautifulSoup(response.content, 'html.parser')

    # Now you can extract text from the linked page using BeautifulSoup methods
    # ... (Your code to extract text using soup)
    # You can use soup.find_all, etc. to navigate and extract relevant text content

    # Example: Extract text from all paragraphs
        text_paragraphs = soup.find_all('p')
        for paragraph in text_paragraphs:
           print(paragraph.text.strip())
    else:
        print(f"Error: Failed to retrieve linked page. Status code: {response.status_code}")

