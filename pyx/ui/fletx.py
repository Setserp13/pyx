import difflib
import flet as ft
import re
import datetime
import calendar
import pyx.mat.mat as mat

def auto_complete(ls, on_select=lambda e: print(e.control.selected_index, e.selection), **kwargs):
    suggestions=[ft.AutoCompleteSuggestion(key=f'{x.lower()} {x.upper()}', value=x) for x in ls]
    return ft.AutoComplete(suggestions=suggestions, on_select=on_select)

# Function to find the closest match
def closest_match(user_input, choices):
    matches = difflib.get_close_matches(user_input, choices, n=1, cutoff=0.1)
    #print(matches)
    return matches[0] if matches else None

def get_close_matches_icase(word, possibilities, n=1, cutoff=0.1, *args, **kwargs):
    """ Case-insensitive version of difflib.get_close_matches """
    lword = word.lower()
    lpos = {p.lower(): p for p in possibilities}
    lmatches = difflib.get_close_matches(lword, lpos.keys(), n=1, cutoff=0.1, *args, **kwargs)
    return lpos[lmatches[0]] if lmatches else None
    #return [lpos[m] for m in lmatches]

class Combobox(ft.Column):
    def __init__(self, page, label, options, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.options = options

        self.on_change = []

        def search_options(e):
            # Filter options based on user input
            #print(e)
            dropdown_container.visible = True
            filtered_options.controls = [
                ft.TextButton(
                    text=option,
                    on_click=lambda e, o=option: select_option(o),
                )
                for option in options if e.control.value.lower() in option.lower()
            ]
            dropdown_container.width = search_field.width
            #for x in self.on_change: x(value)
            page.update()

        def select_option(selected):
            # Set the selected option in the input field and hide the dropdown
            #print(selected)
            #search_field.value = selected
            self.set_value(selected)
            dropdown_container.visible = False
            page.update()

        def on_blur(e):
            # Hide dropdown when TextField loses focus
            if not self.on_container:
                dropdown_container.visible = False
                #search_field.value = get_close_matches_icase(search_field.value, self.options)
                self.set_value(get_close_matches_icase(search_field.value, self.options))
            page.update()

        self.on_container = False

        def on_hover(e): self.on_container = e.data == 'true'

        # TextField for typing and searching
        search_field = ft.TextField(
            label=label,
            hint_text="Type to search...",
            on_change=search_options,
            on_blur=on_blur,  # Hide dropdown on blur
            #width=200,
        )
        self.text_field = search_field

        # Dropdown container
        filtered_options = ft.ListView(
            expand=True,
            spacing=5,  # Space between options
            padding=5,
            height=200,  # Fixed height for the scrollable area
        )
        dropdown_container = ft.Container(
            content=filtered_options,
            visible=False,  # Initially hidden
            border=ft.border.all(1, ft.colors.BLACK),
            bgcolor=ft.colors.WHITE,
            padding=5,
            on_hover=on_hover,
            shadow=ft.BoxShadow(blur_radius=5, color=ft.colors.GREY),
        )
        self.controls = [search_field, dropdown_container]

    def set_value(self, value):
        if self.text_field.value != value:
            for x in self.on_change: x(value)
            self.text_field.value = value

def findallint(text):
    integers = re.findall(r'-?\d+', text)
    return list(map(int, integers))

class IntField(ft.Row):
    def __init__(self, page, label, min=float('-inf'), max=float('inf'), value=0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def autocorrect(e):
            self.text_field.value = mat.clamp(int(0 if self.text_field.value == '' else self.text_field.value), min, max)
            page.update()

        self.text_field = ft.TextField(label=label, input_filter=ft.NumbersOnlyInputFilter(), keyboard_type=ft.KeyboardType.NUMBER, on_blur=autocorrect, expand=True, value=value)

        def increment(e, amount=1):
            self.text_field.value = int(self.text_field.value) + amount
            autocorrect(e)

        remove = ft.IconButton(icon=ft.icons.REMOVE, on_click=lambda e: increment(e, -1))

        add = ft.IconButton(icon=ft.icons.ADD, on_click=lambda e: increment(e))

        self.controls += [self.text_field, remove, add]

class DateField(ft.Row):
    def __init__(self, page, label, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def on_blur(e):
            today = datetime.date.today()
            day, month, year = today.day, today.month, today.year
            integers = findallint(self.text_field.value)
            if len(integers) > 2:
                y = str(integers[2])
                y = y[:min(4, len(y))]
                year = int(str(today.year)[:4-len(y)] + y)
            if len(integers) > 1:
                month = mat.clamp(integers[1], 1, 12)
            if len(integers) > 0:
                _, days_in_month = calendar.monthrange(year, month)
                day = mat.clamp(integers[0], 1, days_in_month)
            self.text_field.value = datetime.date(day=day, month=month, year=year).strftime('%d/%m/%Y')
            page.update()

        self.text_field = ft.TextField(label=label, keyboard_type=ft.KeyboardType.DATETIME, on_blur=on_blur, expand=True)
        self.text_field.value = datetime.date.today().strftime('%d/%m/%Y')

        def handle_change(e):
            self.text_field.value = e.control.value.strftime('%d/%m/%Y')
            page.update()

        def handle_dismissal(e): print("DatePicker dismissed")

        date = ft.IconButton(
            icon=ft.icons.CALENDAR_MONTH,
            on_click=lambda e: page.open(
                ft.DatePicker(
                    #first_date=datetime.datetime(year=2023, month=10, day=1),
                    #last_date=datetime.datetime(year=2024, month=10, day=1),
                    on_change=handle_change,
                    on_dismiss=handle_dismissal,
                )
            ),
        )

        self.controls.append(self.text_field)   
        self.controls.append(date)
