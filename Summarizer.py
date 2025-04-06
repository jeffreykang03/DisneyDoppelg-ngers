#ML
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import wikipedia
import wikipediaapi
import requests
from bs4 import BeautifulSoup
import time
import spacy

wikipedia.set_lang("en")

def get_wiki(title):
    wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='PersonalityTraitsExtractor/1.0 (contact: c7tu@ucsd.edu)'
    )  
    page = wiki.page(title)

    if not page.exists():
        return ""
    # Include summary + 1–2 relevant sections
    content = page.summary.strip()
    keywords = ["character", "personality", "traits", "characteristics", "overview"]
    
    for section in page.sections:
        if any(k in section.title.lower() for k in keywords):
            content += "\n" + section.text.strip()
    return content


BASE_URL = "https://disney.fandom.com/wiki/"
nlp = spacy.load("en_core_web_sm")

df =  pd.read_csv('/Users/claire/Desktop/DF4.csv')

def get_character_links(lst):
    """
    Gather all character page links from a category page.
    """
    links = []
    
    for name in lst:
        name = name.replace(' ', '_')
        links.append(BASE_URL + name)
    return links

def scrape_all_text(character_url):
    response = requests.get(character_url)
    full_text = ""
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        sections = soup.find_all(["h2", "h3"])

        important_sections = ["personality", "biography", "background", "character information"]
        content = []
        for section in sections:
            title = section.get_text().lower()
            if any(imp in title for imp in important_sections):
                sibling = section.find_next_sibling()
                while sibling and sibling.name not in ["h2", "h3"]:
                    if sibling.name == "p":
                        content.append(sibling.get_text(strip=True))
                    sibling = sibling.find_next_sibling()
        if content:
            full_text = " ".join(content)
        else:
            # fallback to all paragraphs (like you're doing now)
            paragraphs = soup.find_all("p")
            all_paragraphs = [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]
            full_text = " ".join(all_paragraphs)
    return full_text

def scrape_personalitysumm_section(character_url):
    response = requests.get(character_url)
    text = ""
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Look for an <h2> or <h3> heading named "Personality" or similar
        headers = soup.find_all(["h2", "h3"])
        aside = soup.find("aside", class_="portable-infobox")
        if aside:
            text_blocks = aside.find_all("div", recursive=True)
            summary_lines = [div.get_text(strip=True) for div in text_blocks if div.get_text(strip=True)]
            text = " ".join(summary_lines)
            print(summary_lines[:10])
        for hdr in headers:
            if "Personality" in hdr.get_text():
                
                personality_section = []
                sibling = hdr.find_next_sibling()
                while sibling and sibling.name not in ["h2", "h3"]:
                    if sibling.name == "p":
                        personality_section.append(sibling.get_text(strip=True))
                    sibling = sibling.find_next_sibling()
                print(personality_section[:10])
                text = " ".join(personality_section)
                break  
    return text


tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

def summarize(text, max_input=512, max_output=100):
    input_text = text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=max_input, truncation=True)
    summary_ids = model.generate(inputs,
        max_length=max_output,min_length=20,length_penalty=2.0,num_beams=4,early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def personality_bool(character_url):
    """
    Scrapes the 'Personality' section (if it exists) from a character's page.
    Returns text of that section.
    """
    response = requests.get(character_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")

        headers = soup.find_all(["h2", "h3"])
        for hdr in headers:
            if "Personality" in hdr.get_text():

                sibling = hdr.find_next_sibling()
                while sibling and sibling.name not in ["h2", "h3"]:
                    if sibling.name == "p":
                        return True
                    sibling = sibling.find_next_sibling()

                break  # We found the "Personality" section, so stop scanning

    return False

def get_character_image(character_url):
    """
    Extracts the main character image from the Disney Wiki page.
    Returns the image URL if found, otherwise an empty string.
    """
    response = requests.get(character_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")

        # Typical Fandom infobox image is within an <aside> or <figure> inside <a><img>
        infobox = soup.find("aside", class_="portable-infobox")
        if infobox:
            img_tag = infobox.find("img")
            if img_tag and img_tag.has_attr("src"):
                return img_tag["src"]
    return ""


def main(characters):
    output = []
    names = []
    for char in characters:
        wiki_text = get_wiki(char)
        if not wiki_text.strip():
            print(f"{char}: [No usable Wikipedia content found]\n")
            continue
        summary = summarize(wiki_text)
        names.append(char)
        output.append(summary)
    return output

def main_fandom(character_names):
    
    raw_links = get_character_links(character_names)
    results_image = []
    results_link = []
    results_summ = []
    # results_traits = []
    # results_name = []

    for link in raw_links:
        # Convert name to wiki-safe format
        # safe_name = name.replace(" ", "_")
        # url = f"{BASE_URL}/wiki/{safe_name}"
        
        # Disney URL alter
        if not personality_bool(link):
            fallback_url = f"{link}_(character)"
            all_text = scrape_all_text(fallback_url)
            link = fallback_url
        else:
            all_text = scrape_all_text(link)
        # all_text = scrape_all_text(link)

        print(link)
        summ = summarize(all_text)

        image_url = get_character_image(link)
        results_image.append(image_url)

        results_summ.append(summ)
        results_link.append(link)
        time.sleep(1)  
        print(summ)
    
    df = pd.DataFrame().assign(character = character_names, dicription = results_summ, image = results_image, link = results_link)

    return df
wiki_titles = ['Mickey Mouse', 'Minnie Mouse', 'Donald Duck', 'Goofy', 'Pluto', 'Daisy Duck', 'Simba', 'Nala', 'Mufasa', 'Scar', 'Timon', 'Pumbaa', 'Elsa', 'Anna', 'Olaf', 'Kristoff', 'Ariel', 'Sebastian', 'Flounder', 'Ursula', 'Belle', 'Beast', 'Lumière', 'Cogsworth', 'Gaston', 'Snow White', 'The Evil Queen', 'Cinderella', 'Fairy Godmother', 'Lady Tremaine', 'Aurora', 'Maleficent', 'Tiana', 'Naveen', 'Dr. Facilier', 'Rapunzel', 'Flynn Rider', 'Pascal', 'Maximus', 'Moana', 'Maui', 'Hei Hei', 'Pocahontas', 'Meeko', 'Mulan', 'Mushu', 'Li Shang', 'Aladdin', 'Jasmine', 'Genie', 'Jafar', 'Abu', 'Rajah', 'Peter Pan', 'Tinker Bell', 'Captain Hook', 'Wendy', 'Dumbo', 'Timothy Q. Mouse', 'Bambi', 'Thumper', 'Flower', 'Pinocchio', 'Jiminy Cricket', 'Figaro', 'Alice', 'The Mad Hatter', 'The Cheshire Cat', 'The Queen of Hearts', 'Winnie the Pooh', 'Piglet', 'Tigger', 'Eeyore', 'Roo', 'Christopher Robin', 'Hercules', 'Megara', 'Hades', 'Phil', 'Tarzan', 'Jane Porter', 'Kala', 'Stitch', 'Lilo', 'Nani', 'Jumba', 'Lightning McQueen', 'Mater', 'Woody', 'Buzz Lightyear', 'Jessie', 'Bo Peep', 'Rex', 'Mike Wazowski', 'Sulley', 'Boo', 'WALL-E', 'EVE', 'Remy', 'Joy'][90:]

avengers_links = [
    "Iron_Man_(Anthony_Stark)",
    "Captain_America_(Steven_Rogers)",
    "Thor_Odinson_(Earth-616)",
    "Hulk_(Bruce_Banner)",
    "Black_Widow_(Natasha_Romanoff)",
    "Hawkeye_(Clint_Barton)",
    "Scarlet_Witch_(Wanda_Maximoff)",
    "Vision_(Earth-616)",
    "Spider-Man_(Peter_Parker)",
    "Doctor_Strange_(Stephen_Strange)"
]

new_disney_names = [   "Baymax", "Hiro Hamada", "Wasabi", "Honey Lemon", "Fred (Big Hero 6)",
    "Go Go Tomago", "King Triton", "Scuttle", "Flotsam and Jetsam", "Vanessa (The Little Mermaid)",
    "Queen Narissa", "Giselle", "Prince Edward", "Nancy Tremaine", "Clarabelle Cow",
    "Horace Horsecollar", "Figaro (Disney)", "Clarice (Disney)", "José Carioca", "Panchito Pistoles",
    "Launchpad McQuack", "Darkwing Duck", "Webby Vanderquack", "Ludwig Von Drake", "Max Goof",
    "Roxanne (A Goofy Movie)", "Powerline (A Goofy Movie)", "Scrooge McDuck", "Huey, Dewey, and Louie", "Pete (Disney)",
    "Gadget Hackwrench", "Zummi Gummi", "Cavin (Gummi Bears)", "Cubbi Gummi", "Tummi Gummi",
    "Grammi Gummi", "Sunni Gummi", "King Gregor", "Queen Elinor (Brave)", "Merida",
    "Anger (Inside Out)", "Joy (Inside Out)", "Sadness (Inside Out)", "Disgust (Inside Out)", "Fear (Inside Out)",
    "Bo Peep", "Jessie (Toy Story)", "Lotso", "Forky", "Zurg"]

disney_names = [
    "Mickey Mouse", "Minnie Mouse", "Donald Duck", "Goofy", "Pluto",
    "Daisy Duck", "Elsa", "Anna", "Olaf", "Kristoff",

    "Simba", "Mufasa", "Nala", "Timon", "Pumbaa",
    "Ariel", "Flounder", "Sebastian", "Belle", "Beast",
    "Gaston", "Cinderella", "Prince Charming", "Snow White", "The Evil Queen",
    "Aurora", "Maleficent", "Aladdin", "Jasmine", "Genie",
    "Abu", "Remy", "Tiana", "Prince Naveen", "Rapunzel",
    "Flynn Rider", "Moana", "Maui", "Stitch", "Lilo",
    "Hercules" , "Megara", "Tarzan", "Jane Porter", "Pocahontas",
    "Fa Mulan", "Mushu", "Wreck-It Ralph", "Vanellope von Schweetz", "Buzz Lightyear", "Woody"]

df = main_fandom(disney_names)
df.to_csv('DF.csv', index=True)
# scrape_personalitysumm_section("https://marvel.fandom.com/wiki/Anthony_Stark_(Earth-616)#History")