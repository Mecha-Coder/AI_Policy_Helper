import Chat from "@/components/Chat";
import AdminPanel from "@/components/AdminPanel";

export default function Page() {
    return (
        <div className="min-h-screen bg-gray-100 pt-12 px-4 sm:px-6 lg:px-16">
            <div className="w-full mx-auto">
                {/* Header */}
                <div className="text-center mb-12">
                    <h1 className="text-4xl font-bold text-gray-900 mb-4">
                        AI Policy & Product Helper
                    </h1>
                    <p className="text-xl text-gray-600 max-w-3xl mx-auto">
                        Local-first RAG starter. Ingest sample docs, ask
                        questions, and see citations.
                    </p>
                </div>

                {/* Main Content Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-12">
                    {/* Left Column - Admin Panel */}
                    <div className="lg:col-span-1">
                        <AdminPanel />
                    </div>

                    {/* Right Column - Chat and Instructions */}
                    <div className="lg:col-span-2 space-y-8">
                        <Chat />

                        {/* How to Test Section */}
                        <div className="bg-white rounded-lg border border-gray-200 p-6 shadow-sm">
                            <h3 className="text-xl font-semibold text-gray-800 mb-4">
                                How to test
                            </h3>
                            <ol className="list-decimal list-inside text-gray-700">
                                <li>
                                    Click <b>Ingest sample docs</b>.
                                </li>
                                <li>
                                    Ask:{" "}
                                    <i>
                                        Can a customer return a damaged blender
                                        after 20 days?
                                    </i>
                                </li>
                                <li>
                                    Ask:{" "}
                                    <i>
                                        What's the shipping SLA to East Malaysia
                                        for bulky items?
                                    </i>
                                </li>
                                <li>
                                    Ask: <i>What couriers use for shipping?</i>
                                </li>
                                <li>
                                    Ask:{" "}
                                    <i>
                                        What is the warranty period for
                                        PowerBlend 100?
                                    </i>
                                </li>
                            </ol>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
