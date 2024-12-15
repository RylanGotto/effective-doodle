
imports.gi.versions.Soup = "3.0";
imports.gi.versions.Gtk = "3.0";
const {Soup, Gtk, Gio} = imports.gi;
const GLib = imports.gi.GLib;

let initMessages = [{ role: "assistant", content: "You are a helpful assistant, help me with any everything.", }];


function readResponse2(stream) {
    let decoder = new TextDecoder('utf-8');
    stream.read_line_async(0, null, (stream, res) => {
        try {
            if (!stream) return;
            const [bytes] = stream.read_line_finish(res);
            const line = decoder.decode(bytes);

            if (line && line.trim() !== '') {
                let data = line.substr(6);
                if (data === '[DONE]') {
                    return;  // End of stream signal
                }

                try {
                    let result = JSON.parse(data);
                    if (result.choices[0].finish_reason === 'stop') {
                        console.log(result.choices[0].finish_reason);
                        return;  // End the stream
                    }
                    GLib.stdout.write(result.choices[0].delta.content);
                    GLib.stdout.flush();
                } catch (err) {
                    console.error("Error parsing JSON or emitting response:", err);
                }
            }
            // Continue processing the next line
            readResponse2(stream)
        } catch (err) {
            console.error("Error reading response:", err);
        }
    });
}

function makeRequest(msg) {
        let session = new Soup.Session();
        let initMessages = [{ role: 'user', content: 'get news on 24 election' }];
        // Ensure initMessages is not overwritten with a string
        const body = {
            model: "gpt-4o",
            messages: initMessages,
            temperature: 0,
            stream: true,
        };
        let key = ''

        let url = 'http://localhost:8000/chat/'
        let message = Soup.Message.new('POST', url);
        

        message.request_headers.append('Authorization', `Bearer ${key}`);
        message.set_request_body_from_bytes('application/json', new GLib.Bytes((JSON.stringify(body))));
        
        session.send_async(message, GLib.DEFAULT_PRIORITY, null, (_, result) => {
            const stream = session.send_finish(result);
            readResponse2(new Gio.DataInputStream({
                close_base_stream: true,
                base_stream: stream
            }));
        });
}

makeRequest('hello')

Gtk.init([]);
Gtk.main();