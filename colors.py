colors_humans = {'Large C': '#8931EF',
                 'Large NC': '#cfa9fc',
                 'Medium C': '#FF8600',
                 'Medium NC': '#fab76e',
                 'Small': '#000000',
                 'XL': '#9400D3',
                 'L': '#0000FF',
                 'M': '#00FF00',
                 'S (> 1)': '#FF7F00',
                 'S': '#FF7F00',
                 'XS': '#FF0000',
                 'Single (1)': '#C77DF3',
                 'Small sim': '#d41e1e',
                 'Medium sim': '#61d41e',
                 'Large sim': '#1ed4d1'
                 }

colors_state = {'b': '#9700fc', 'be': '#e0c1f5', 'b1': '#d108ba', 'b2': '#38045c',
                'b1/b2': '#d108ba',
                # 'a': '#fc0000',
                'ac': '#fc0000', 'ab': '#802424',
                'c': '#fc8600', 'cg': '#8a4b03',
                'e': '#fcf400', 'eb': '#a6a103', 'eg': '#05f521',
                'f': '#30a103', 'g': '#000000',
                'h': '#085cd1', False: '#000000', True: '#ccffcc', 'i': '#000000', }



def hex_to_rgb(hex_str, norm=1):
    hex_str = hex_str.lstrip('#')
    h_len = len(hex_str)
    return tuple(int(hex_str[i:i+h_len//3], 16)/255 for i in range(0, h_len, h_len//3))

def hex_to_rgba(hex_str, alpha=1.0):
    hex_str = hex_str.lstrip('#')
    h_len = len(hex_str)
    return tuple(int(hex_str[i:i+h_len//4], 16) for i in range(0, h_len, h_len//4)) + (alpha,)
