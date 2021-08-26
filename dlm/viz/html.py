import os
from datetime import datetime

class HTMLGenerator:
    def __init__(self, path=None, title=None, subtitle=None, navigate = False):
        
        self.here = os.path.dirname(__file__)
        
        self.path = path
        self.title = title
        self.subtitle = subtitle if subtitle is not None else datetime.now().strftime("%d/%m/%Y - %H:%M:%S")
        self.navigate = navigate
        self.header = []
        self.body = []
        
        self.toc = []
        
        self.make_header()
        self.make_body()
        
    def make_header(self):
        self.header.append(open(os.path.join(self.here, "html/header.html"), "r").read())
        self.header[-1].replace("<title></title>", f"<title>{self.title}</title>")
        
        if self.navigate:
            self.header.append(open(os.path.join(self.here, "html/menu.html"), "r").read())
            
        self.body.append(f'<section class="page-header"><h1 class="project-name">{self.title}</h1><h2 class="project-tagline">{self.subtitle}</h2></section>')
        
    def make_body(self):
        self.body.append('<section class="main-content">')
        self.body.append('<body>\n')        
        
    def add_to_body(self, title = None, content = None):
        if title is not None:
            self.add_title(1, title)
        if content is not None:
            self.body.append(f'<div class="blank">{content}</div>\n')
        
    def add_title(self, level, title_content):
        assert level in [1, 2, 3], "Level should be in [1, 2, 3]"
        body = []
        if level == 1:
            body.append('<center>\n')
        body.append(f'\t<h{level} id="{title_content}">\n')
        body.append(f'\t\t{title_content}\n')
        body.append(f'\t</h{level}>\n')
        if level == 1:
            body.append('</center>\n')
        self.body.append("".join(body))
        
        self.toc.append(f"<li><a href='#{title_content}'>{title_content}</a></li>\n")
        
    def input_toc(self):
        body_str = self.body_str.split("<body>\n")
        toc = ["<body>\n"]
        toc.append(f'\t<h{2}>\n')
        toc.append(f'\t\tTable of contents\n')
        toc.append(f'\t</h{2}>\n')
        toc.append("<div>\n")
        toc.append("<ul>\n")
        toc.append("".join(self.toc))
        toc.append("</ul>\n")
        toc.append("</div>\n")
        
        self.body_str = ("".join(toc)).join(body_str)
        
        
    def return_html(self):
        begin_html = '<!DOCTYPE html>\n<html>\n'
        self.header_str = "".join(self.header)
        self.body_str = "".join(self.body)
        end_html = "</html>\n"
        
        self.input_toc()
        
        webpage = begin_html + self.header_str + self.body_str + open(os.path.join(self.here, "html/footer.html"), "r").read() + '</body>\n</section>\n' + end_html
        
        if self.path is not None:
            with open(self.path, 'w') as output_file:
                output_file.write(webpage)
                
        return webpage