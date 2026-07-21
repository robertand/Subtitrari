import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
from transformers import MarianMTModel, MarianTokenizer

class TraducatorTR_RO:
    def __init__(self, root):
        self.root = root
        self.root.title("Traducător Turcă → Română (Local)")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Variabile pentru model
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        
        # Setează stilul
        self.root.configure(bg='#f0f0f0')
        
        self.create_widgets()
        self.load_model_in_background()
    
    def create_widgets(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Status bar
        self.status_label = ttk.Label(main_frame, text="🔄 Se încarcă modelul... (aproximativ 300MB)", 
                                      font=('Arial', 10, 'italic'))
        self.status_label.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Label text sursă
        ttk.Label(main_frame, text="📝 Text în Turcă:", font=('Arial', 12, 'bold')).grid(
            row=1, column=0, sticky=tk.W, pady=(0, 5))
        
        # Buton swap limbi (nu e funcțional complet, dar pentru design)
        swap_btn = ttk.Button(main_frame, text="⇄", width=3, command=self.swap_text)
        swap_btn.grid(row=1, column=1, sticky=tk.E, pady=(0, 5))
        
        # Zona de text pentru turcă
        self.text_turkish = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, 
                                                      height=8, font=('Arial', 11))
        self.text_turkish.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Buton traducere
        self.translate_btn = ttk.Button(main_frame, text="🌐 Tradu în Română", 
                                        command=self.translate_text, state='disabled')
        self.translate_btn.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Label text țintă
        ttk.Label(main_frame, text="🎯 Traducere în Română:", font=('Arial', 12, 'bold')).grid(
            row=4, column=0, sticky=tk.W, pady=(0, 5))
        
        # Buton copiere
        copy_btn = ttk.Button(main_frame, text="📋 Copiază", command=self.copy_translation)
        copy_btn.grid(row=4, column=1, sticky=tk.E, pady=(0, 5))
        
        # Zona de text pentru română
        self.text_romanian = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, 
                                                       height=8, font=('Arial', 11), 
                                                       state='normal')
        self.text_romanian.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def load_model_in_background(self):
        """Încarcă modelul într-un thread separat pentru a nu bloca interfața"""
        thread = threading.Thread(target=self.load_model, daemon=True)
        thread.start()
    
    def load_model(self):
        """Încarcă modelul MarianMT"""
        try:
            model_name = "Helsinki-NLP/opus-mt-tc-base-tr-ro"
            
            self.update_status("📥 Se descarcă modelul... (prima pornire poate dura 2-3 minute)", False)
            self.start_progress()
            
            # Încarcă tokenizer și model
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name)
            
            self.model_loaded = True
            self.stop_progress()
            self.update_status("✅ Model încărcat cu succes! Poți începe să traduci.", True)
            
            # Activează butonul de traducere
            self.root.after(0, lambda: self.translate_btn.config(state='normal'))
            
        except Exception as e:
            self.stop_progress()
            self.update_status(f"❌ Eroare la încărcarea modelului: {str(e)}", False)
            messagebox.showerror("Eroare", f"Nu s-a putut încărca modelul:\n{str(e)}")
    
    def translate_text(self):
        """Traduce textul din turcă în română"""
        if not self.model_loaded:
            messagebox.showwarning("Atenție", "Modelul nu este încă încărcat. Așteaptă...")
            return
        
        turkish_text = self.text_turkish.get("1.0", tk.END).strip()
        
        if not turkish_text:
            messagebox.showinfo("Info", "Te rog să introduci un text în turcă pentru traducere.")
            return
        
        # Dezactivează butonul în timpul traducerii
        self.translate_btn.config(state='disabled')
        self.start_progress()
        self.update_status("🔄 Se traduce...", False)
        
        # Rulează traducerea într-un thread separat
        thread = threading.Thread(target=self.perform_translation, args=(turkish_text,), daemon=True)
        thread.start()
    
    def perform_translation(self, turkish_text):
        """Execută traducerea efectivă"""
        try:
            # Tokenizare
            batch = self.tokenizer([turkish_text], return_tensors="pt", 
                                   padding=True, truncation=True, max_length=512)
            
            # Generare traducere
            generated_ids = self.model.generate(**batch)
            
            # Decodificare
            translated = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Actualizează interfața
            self.root.after(0, lambda: self.display_translation(translated))
            self.root.after(0, lambda: self.update_status("✅ Traducere completată!", True))
            
        except Exception as e:
            self.root.after(0, lambda: self.update_status(f"❌ Eroare la traducere: {str(e)}", False))
            self.root.after(0, lambda: messagebox.showerror("Eroare", f"Traducerea a eșuat:\n{str(e)}"))
        finally:
            self.root.after(0, lambda: self.stop_progress())
            self.root.after(0, lambda: self.translate_btn.config(state='normal'))
    
    def display_translation(self, text):
        """Afișează traducerea în zona de text"""
        self.text_romanian.delete("1.0", tk.END)
        self.text_romanian.insert("1.0", text)
    
    def copy_translation(self):
        """Copiază traducerea în clipboard"""
        translation = self.text_romanian.get("1.0", tk.END).strip()
        if translation:
            self.root.clipboard_clear()
            self.root.clipboard_append(translation)
            self.update_status("📋 Traducerea a fost copiată în clipboard!", False)
            # Revine la statusul normal după 2 secunde
            self.root.after(2000, lambda: self.update_status("✅ Model încărcat", True))
    
    def swap_text(self):
        """Schimbă textul între cele două casete (simbolic, modelul e doar TR->RO)"""
        turkish = self.text_turkish.get("1.0", tk.END).strip()
        romanian = self.text_romanian.get("1.0", tk.END).strip()
        if romanian and not turkish:
            self.text_turkish.delete("1.0", tk.END)
            self.text_turkish.insert("1.0", romanian)
            self.update_status("ℹ️ Notă: Modelul traduce doar din Turcă în Română", False)
    
    def update_status(self, message, is_permanent=False):
        """Actualizează bara de status"""
        self.status_label.config(text=message)
        if is_permanent:
            # După 3 secunde revine la normal dacă e permanent
            self.root.after(3000, lambda: self.status_label.config(text="✅ Model gata de utilizare" 
                                                                    if self.model_loaded else "🔄 Model încărcat"))
    
    def start_progress(self):
        """Pornește animation progress bar"""
        self.progress.start(10)
    
    def stop_progress(self):
        """Oprește animation progress bar"""
        self.progress.stop()

def main():
    root = tk.Tk()
    app = TraducatorTR_RO(root)
    root.mainloop()

if __name__ == "__main__":
    main()