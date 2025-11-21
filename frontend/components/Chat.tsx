"use client";
import React from "react";
import { apiAsk } from "@/lib/api";

type Message = {
    role: "user" | "assistant";
    content: string;
    citations?: { title: string; section?: string }[];
    chunks?: { title: string; section?: string; text: string }[];
};

export default function Chat() {
    const [messages, setMessages] = React.useState<Message[]>([]);
    const [q, setQ] = React.useState("");
    const [loading, setLoading] = React.useState(false);

    const send = async () => {
        if (!q.trim()) return;
        const my = { role: "user" as const, content: q };
        setMessages((m) => [...m, my]);
        setLoading(true);
        try {
            const res = await apiAsk(q);
            const ai: Message = {
                role: "assistant",
                content: res.answer,
                citations: res.citations,
                chunks: res.chunks,
            };
            setMessages((m) => [...m, ai]);
        } catch (e: any) {
            setMessages((m) => [
                ...m,
                { role: "assistant", content: "Error: " + e.message },
            ]);
        } finally {
            setLoading(false);
            setQ("");
        }
    };

    return (
        <div className="bg-white rounded-lg border border-gray-200 p-6 shadow-sm">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Chat</h2>

            <div className="h-96 overflow-y-auto border border-gray-200 rounded-lg p-4 mb-4 bg-gray-50">
                {messages.map((m, i) => (
                    <div key={i} className="mb-4 last:mb-0">
                        <div className="text-sm font-medium text-gray-500 mb-1">
                            {m.role === "user" ? "You" : "Assistant"}
                        </div>
                        <div className="text-gray-700">{m.content}</div>

                        {m.citations && m.citations.length > 0 && (
                            <div className="flex flex-wrap gap-2 mt-2">
                                {m.citations.map((c, idx) => (
                                    <span
                                        key={idx}
                                        className="inline-block px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded border border-blue-200"
                                        title={c.section || ""}
                                    >
                                        {c.title}
                                    </span>
                                ))}
                            </div>
                        )}

                        {m.chunks && m.chunks.length > 0 && (
                            <details className="mt-3">
                                <summary className="cursor-pointer text-sm text-gray-600 hover:text-gray-800 font-medium">
                                    View supporting chunks
                                </summary>
                                <div className="mt-2 space-y-3">
                                    {m.chunks.map((c, idx) => (
                                        <div
                                            key={idx}
                                            className="bg-white border border-gray-200 rounded-lg p-3 text-sm"
                                        >
                                            <div className="font-semibold text-gray-800 mb-1">
                                                {c.title}
                                                {c.section
                                                    ? " — " + c.section
                                                    : ""}
                                            </div>
                                            <div className="text-gray-600 whitespace-pre-wrap">
                                                {c.text}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </details>
                        )}
                    </div>
                ))}
            </div>

            <div className="flex flex-col sm:flex-row gap-3">
                <input
                    type="text"
                    placeholder="Ask about policy or products..."
                    value={q}
                    onChange={(e) => setQ(e.target.value)}
                    onKeyDown={(e) => {
                        if (e.key === "Enter") send();
                    }}
                    className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-colors"
                />
                <button
                    onClick={send}
                    disabled={loading}
                    className="px-6 py-2 bg-gray-900 text-white rounded-lg hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
                >
                    {loading ? "Thinking..." : "Send"}
                </button>
            </div>
        </div>
    );
}
